package rag

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/ai-engine/go/internal/config"
	pb "github.com/ai-engine/proto/go"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type Manager struct {
	mu           sync.RWMutex
	documents    map[string]*Document
	config       *config.Config
	chunkSize    int
	chunkOverlap int
	topK         int
}

type Service interface {
	UpsertDocument(context.Context, *pb.UpsertRequest) (*pb.UpsertResponse, error)
	DeleteDocument(context.Context, *pb.DeleteRequest) (*emptypb.Empty, error)
	Search(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error)
	GetRagStatus(context.Context, *emptypb.Empty) (*pb.RagStatus, error)
	ListDocuments(context.Context, *emptypb.Empty) (*pb.DocumentList, error)
	DocumentCount() int64
}

type Document struct {
	ID        string
	Title     string
	Content   string
	Metadata  map[string]string
	Chunks    []Chunk
	CreatedAt int64
	UpdatedAt int64
}

type Chunk struct {
	ID         string
	DocumentID string
	Text       string
	Index      int
	Metadata   map[string]string
}

func NewManager(cfg *config.Config) *Manager {
	return &Manager{
		documents:    make(map[string]*Document),
		config:       cfg,
		chunkSize:    cfg.RAG.ChunkSize,
		chunkOverlap: cfg.RAG.ChunkOverlap,
		topK:         cfg.RAG.TopK,
	}
}

func (m *Manager) UpsertDocument(ctx context.Context, req *pb.UpsertRequest) (*pb.UpsertResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Simple chunking (in production, use the Rust chunking crate)
	chunks := m.chunkText(req.Content)

	doc := &Document{
		ID:        req.DocumentId,
		Content:   req.Content,
		Metadata:  req.Metadata,
		Chunks:    make([]Chunk, len(chunks)),
		CreatedAt: now(),
		UpdatedAt: now(),
	}

	// Extract title from metadata or use first line
	if title, ok := req.Metadata["title"]; ok {
		doc.Title = title
	} else {
		doc.Title = req.DocumentId
	}

	for i, chunkText := range chunks {
		doc.Chunks[i] = Chunk{
			ID:         fmt.Sprintf("%s-chunk-%d", req.DocumentId, i),
			DocumentID: req.DocumentId,
			Text:       chunkText,
			Index:      i,
		}
	}

	m.documents[req.DocumentId] = doc

	return &pb.UpsertResponse{
		DocumentId:    req.DocumentId,
		ChunksIndexed: int32(len(chunks)),
	}, nil
}

func (m *Manager) chunkText(text string) []string {
	if len(text) == 0 {
		return nil
	}

	var chunks []string
	chars := []rune(text)
	size := m.chunkSize

	for i := 0; i < len(chars); i += size - m.chunkOverlap {
		end := i + size
		if end > len(chars) {
			end = len(chars)
		}
		chunk := string(chars[i:end])
		if len(chunk) > 0 {
			chunks = append(chunks, chunk)
		}
		if end >= len(chars) {
			break
		}
	}

	return chunks
}

func (m *Manager) DeleteDocument(ctx context.Context, req *pb.DeleteRequest) (*emptypb.Empty, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.documents[req.DocumentId]; !exists {
		return nil, fmt.Errorf("document not found: %s", req.DocumentId)
	}

	delete(m.documents, req.DocumentId)

	return &emptypb.Empty{}, nil
}

func (m *Manager) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	topK := int(req.TopK)
	if topK == 0 {
		topK = m.topK
	}

	var results []*pb.SearchResult
	var resultsMu sync.Mutex
	g, ctx := errgroup.WithContext(ctx)

	for _, doc := range m.documents {
		doc := doc
		if len(req.Filters) > 0 {
			match := true
			for k, v := range req.Filters {
				if doc.Metadata[k] != v {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}

		for _, chunk := range doc.Chunks {
			chunk := chunk
			g.Go(func() error {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
					score := m.computeSimilarity(req.Query, chunk.Text)
					if score > 0.1 {
						resultsMu.Lock()
						results = append(results, &pb.SearchResult{
							DocumentId: doc.ID,
							ChunkText:  chunk.Text,
							Score:      score,
							Metadata:   doc.Metadata,
						})
						resultsMu.Unlock()
					}
					return nil
				}
			})
		}
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > topK {
		results = results[:topK]
	}

	return &pb.SearchResponse{
		Results:     results,
		QueryTimeMs: 0.0,
	}, nil
}

func (m *Manager) computeSimilarity(query, text string) float32 {
	// Simplified similarity - in production use cosine similarity on embeddings
	queryLower := toLower(query)
	textLower := toLower(text)

	if textLower == "" {
		return 0
	}

	matches := 0
	words := splitWords(queryLower)
	for _, word := range words {
		if contains(textLower, word) {
			matches++
		}
	}

	return float32(matches) / float32(len(words)+1)
}

func (m *Manager) GetRagStatus(ctx context.Context, _ *emptypb.Empty) (*pb.RagStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	docCount := int64(len(m.documents))
	chunkCount := int64(0)
	for _, doc := range m.documents {
		chunkCount += int64(len(doc.Chunks))
	}

	return &pb.RagStatus{
		DocumentCount:  docCount,
		ChunkCount:     chunkCount,
		IndexSizeBytes: chunkCount * 512, // estimated
		EmbeddingModel: m.config.RAG.EmbeddingModel,
	}, nil
}

func (m *Manager) ListDocuments(ctx context.Context, _ *emptypb.Empty) (*pb.DocumentList, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	docs := make([]*pb.DocumentInfo, 0, len(m.documents))
	for _, doc := range m.documents {
		docs = append(docs, &pb.DocumentInfo{
			Id:         doc.ID,
			Title:      doc.Title,
			ChunkCount: int64(len(doc.Chunks)),
			CreatedAt:  timestamppb.New(time.Unix(doc.CreatedAt, 0)),
			UpdatedAt:  timestamppb.New(time.Unix(doc.UpdatedAt, 0)),
		})
	}

	return &pb.DocumentList{Documents: docs}, nil
}

func now() int64 {
	return time.Now().Unix()
}

func toLower(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		result[i] = c
	}
	return string(result)
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func splitWords(s string) []string {
	var words []string
	var word []rune
	for _, c := range s {
		if c == ' ' || c == '\n' || c == '\t' {
			if len(word) > 0 {
				words = append(words, string(word))
				word = nil
			}
		} else {
			word = append(word, c)
		}
	}
	if len(word) > 0 {
		words = append(words, string(word))
	}
	return words
}

func (m *Manager) DocumentCount() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return int64(len(m.documents))
}
package rag

import (
	"context"
	"strings"
	"testing"

	"github.com/ai-engine/go/internal/config"
	pb "github.com/ai-engine/proto/go"
)

func TestSearchReturnsStableResultsForChunkedDocument(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.RAG.ChunkSize = 16
	cfg.RAG.ChunkOverlap = 4

	manager := NewManager(cfg)
	content := strings.Repeat("vector search text ", 64)
	if _, err := manager.UpsertDocument(context.Background(), &pb.UpsertRequest{
		DocumentId: "doc-1",
		Content:    content,
		Metadata: map[string]string{
			"title":       "Chunked Document",
			"source_db":   "turso",
			"external_id": "entity-1",
		},
	}); err != nil {
		t.Fatalf("upsert document: %v", err)
	}

	response, err := manager.Search(context.Background(), &pb.SearchRequest{
		Query: "vector search",
		TopK:  10,
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(response.Results) == 0 {
		t.Fatal("expected at least one search result")
	}
}

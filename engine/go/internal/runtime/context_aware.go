package runtime

import (
	"context"
	"io"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"
)

type ContextOrchestrationBackend interface {
	Enabled() bool
	GetSession(context.Context, string) (*contextsvc.SessionResponse, error)
	Search(context.Context, contextsvc.SearchRequest) (*contextsvc.SearchResponse, error)
	AppendSession(context.Context, contextsvc.SessionAppendRequest) (*contextsvc.SessionResponse, error)
	UpsertResource(context.Context, contextsvc.UpsertResourceRequest) (*contextsvc.UpsertResourceResponse, error)
}

type ContextAwareService struct {
	inner        Service
	context      ContextOrchestrationBackend
	timeout      time.Duration
	memoryWindow int
	graphTopK    int
	documentTopK int
}

var preferOverPattern = regexp.MustCompile(`(?i)\bprefer(?:s|red)?\s+([a-z0-9._ -]+?)\s+over\s+([a-z0-9._ -]+)`)

// NewContextAwareService creates a ContextAwareService that wraps the provided inner Service
// and uses the given ContextOrchestrationBackend to optionally augment inference requests.
// The returned service is initialized with a 1500ms context assembly timeout, a memory window
// of 8 entries, and top-K values of 4 for both graph and document searches.
func NewContextAwareService(inner Service, contextBackend ContextOrchestrationBackend) *ContextAwareService {
	return &ContextAwareService{
		inner:        inner,
		context:      contextBackend,
		timeout:      1500 * time.Millisecond,
		memoryWindow: 8,
		graphTopK:    4,
		documentTopK: 4,
	}
}

func (s *ContextAwareService) GetStatus(ctx context.Context, req *emptypb.Empty) (*pb.RuntimeStatus, error) {
	return s.inner.GetStatus(ctx, req)
}

func (s *ContextAwareService) ListModels(ctx context.Context, req *emptypb.Empty) (*pb.ModelList, error) {
	return s.inner.ListModels(ctx, req)
}

func (s *ContextAwareService) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.ModelInfo, error) {
	return s.inner.LoadModel(ctx, req)
}

func (s *ContextAwareService) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*emptypb.Empty, error) {
	return s.inner.UnloadModel(ctx, req)
}

func (s *ContextAwareService) LoadedModelCount() int {
	return s.inner.LoadedModelCount()
}

func (s *ContextAwareService) StreamInference(ctx context.Context, stream pb.Runtime_StreamInferenceServer) error {
	if s.context == nil || !s.context.Enabled() {
		return s.inner.StreamInference(ctx, stream)
	}

	group, groupCtx := errgroup.WithContext(ctx)
	requests := make(chan *pb.InferenceRequest)

	var (
		lastSessionID string
		lastPrompt    string
		responseText  strings.Builder
		responseMu    sync.Mutex
	)

	proxy := &contextAwareProxyStream{
		ctx:      groupCtx,
		requests: requests,
		send:     stream.Send,
		onSend: func(resp *pb.InferenceResponse) {
			if resp == nil || resp.GetToken() == "" {
				return
			}
			responseMu.Lock()
			responseText.WriteString(resp.GetToken())
			responseMu.Unlock()
		},
	}

	group.Go(func() error {
		defer close(requests)
		for {
			req, err := stream.Recv()
			if err == io.EOF {
				return nil
			}
			if err != nil {
				return err
			}

			assembled, sessionID := s.assembleRequest(groupCtx, req)
			if sessionID != "" {
				lastSessionID = sessionID
				lastPrompt = assembled.GetPrompt()
				_, _ = s.context.AppendSession(groupCtx, contextsvc.SessionAppendRequest{
					SessionID: sessionID,
					Role:      "user",
					Content:   req.GetPrompt(),
					Metadata: map[string]string{
						"source": "runtime.context-aware",
					},
				})
			}

			select {
			case <-groupCtx.Done():
				return groupCtx.Err()
			case requests <- assembled:
			}
		}
	})

	group.Go(func() error {
		return s.inner.StreamInference(groupCtx, proxy)
	})

	err := group.Wait()
	if lastSessionID != "" {
		responseMu.Lock()
		assistantReply := responseText.String()
		responseMu.Unlock()
		_, _ = s.context.AppendSession(context.Background(), contextsvc.SessionAppendRequest{
			SessionID: lastSessionID,
			Role:      "assistant",
			Content:   assistantReply,
			Metadata: map[string]string{
				"source": "runtime.context-aware",
			},
		})
		s.learnFromTurn(context.Background(), lastSessionID, lastPrompt, assistantReply)
	}
	return err
}

func (s *ContextAwareService) assembleRequest(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceRequest, string) {
	cloned := cloneInferenceRequest(req)
	sessionID := cloned.GetParameters()["session_id"]
	if sessionID == "" {
		return cloned, ""
	}

	assemblyCtx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	var (
		sessionResp *contextsvc.SessionResponse
		graphResp   *contextsvc.SearchResponse
		docResp     *contextsvc.SearchResponse
		sessionErr  error
		graphErr    error
		docErr      error
	)

	group, groupCtx := errgroup.WithContext(assemblyCtx)
	group.Go(func() error {
		sessionResp, sessionErr = s.context.GetSession(groupCtx, sessionID)
		return nil
	})
	group.Go(func() error {
		graphResp, graphErr = s.context.Search(groupCtx, contextsvc.SearchRequest{
			Query: req.GetPrompt(),
			TopK:  s.graphTopK,
			Filters: map[string]string{
				"kind": "graph",
			},
			Layer: contextsvc.LayerL1,
		})
		return nil
	})
	group.Go(func() error {
		docResp, docErr = s.context.Search(groupCtx, contextsvc.SearchRequest{
			Query: req.GetPrompt(),
			TopK:  s.documentTopK,
			Filters: map[string]string{
				"kind": "resource",
			},
			Layer: contextsvc.LayerL2,
		})
		return nil
	})

	if err := group.Wait(); err != nil {
		return cloned, sessionID
	}
	if sessionErr != nil && graphErr != nil && docErr != nil {
		return cloned, sessionID
	}

	sections := make([]string, 0, 4)
	if formatted := formatRecentSession(sessionResp, s.memoryWindow); formatted != "" {
		sections = append(sections, formatted)
	}
	if formatted := formatSearchHits("Graph facts", graphResp); formatted != "" {
		sections = append(sections, formatted)
	}
	if formatted := formatSearchHits("Retrieved documents", docResp); formatted != "" {
		sections = append(sections, formatted)
	}
	if len(sections) == 0 {
		return cloned, sessionID
	}

	sections = append(sections, "User request:\n"+req.GetPrompt())
	cloned.Prompt = strings.Join(sections, "\n\n")
	cloned.ContextRefs = mergeContextRefs(cloned.ContextRefs, graphResp, docResp)
	return cloned, sessionID
}

// formatRecentSession formats the most recent session entries from resp into a multi-line string suitable for inclusion in a prompt.
// If resp is nil or has no entries it returns an empty string.
// It includes at most the last limit entries, prefixed by "Recent working memory:" and each entry on its own line as "- <Role>: <Content>".
// Lines are joined with "\n".
func formatRecentSession(resp *contextsvc.SessionResponse, limit int) string {
	if resp == nil || len(resp.Entries) == 0 {
		return ""
	}
	start := 0
	if len(resp.Entries) > limit {
		start = len(resp.Entries) - limit
	}
	lines := make([]string, 0, len(resp.Entries[start:])+1)
	lines = append(lines, "Recent working memory:")
	for _, entry := range resp.Entries[start:] {
		lines = append(lines, "- "+entry.Role+": "+entry.Content)
	}
	return strings.Join(lines, "\n")
}

// formatSearchHits formats a search response as a titled bullet list.
// If resp is nil or contains no results it returns an empty string.
// The output begins with `title + ":"` on its own line, followed by one line per
// result in the form `- <ChunkText>`, joined with newline characters.
func formatSearchHits(title string, resp *contextsvc.SearchResponse) string {
	if resp == nil || len(resp.Results) == 0 {
		return ""
	}
	lines := []string{title + ":"}
	for _, result := range resp.Results {
		lines = append(lines, "- "+result.ChunkText)
	}
	return strings.Join(lines, "\n")
}

// before any newly discovered refs from the responses.
func mergeContextRefs(existing []string, responses ...*contextsvc.SearchResponse) []string {
	seen := make(map[string]struct{}, len(existing))
	merged := make([]string, 0, len(existing)+4)
	for _, ref := range existing {
		if ref == "" {
			continue
		}
		if _, ok := seen[ref]; ok {
			continue
		}
		seen[ref] = struct{}{}
		merged = append(merged, ref)
	}
	for _, resp := range responses {
		if resp == nil {
			continue
		}
		for _, result := range resp.Results {
			if result.URI == "" {
				continue
			}
			if _, ok := seen[result.URI]; ok {
				continue
			}
			seen[result.URI] = struct{}{}
			merged = append(merged, result.URI)
		}
	}
	return merged
}

// cloneInferenceRequest creates a shallow clone of the provided InferenceRequest.
// If req is nil, it returns an empty InferenceRequest. The returned request has
// newly allocated copies of the Parameters map and ContextRefs slice to avoid
// sharing those containers with the original; other scalar fields are copied.
func cloneInferenceRequest(req *pb.InferenceRequest) *pb.InferenceRequest {
	if req == nil {
		return &pb.InferenceRequest{}
	}
	parameters := make(map[string]string, len(req.GetParameters()))
	for key, value := range req.GetParameters() {
		parameters[key] = value
	}
	contextRefs := append([]string(nil), req.GetContextRefs()...)
	return &pb.InferenceRequest{
		ModelId:     req.GetModelId(),
		Prompt:      req.GetPrompt(),
		Parameters:  parameters,
		Provider:    req.GetProvider(),
		ContextRefs: contextRefs,
	}
}

func (s *ContextAwareService) learnFromTurn(ctx context.Context, sessionID, prompt, assistantReply string) {
	if s.context == nil || sessionID == "" || assistantReply == "" {
		return
	}

	var fact *contextsvc.UpsertResourceRequest
	combined := strings.ToLower(strings.TrimSpace(prompt + "\n" + assistantReply))
	if matches := preferOverPattern.FindStringSubmatch(combined); len(matches) == 3 {
		preferred := normalizeFactID(matches[1])
		if preferred != "" {
			fact = buildPreferenceFact(sessionID, preferred)
		}
	} else if strings.Contains(combined, "dragonfly") &&
		strings.Contains(combined, "prefer") {
		fact = buildPreferenceFact(sessionID, "dragonfly")
	} else if strings.Contains(combined, "dragonfly") &&
		strings.Contains(combined, "working memory") {
		fact = buildPreferenceFact(sessionID, "dragonfly")
	}

	if fact == nil {
		return
	}
	_, _ = s.context.UpsertResource(ctx, *fact)
}

// buildPreferenceFact constructs an UpsertResourceRequest representing a user preference
// fact for the specified session and object identifier. The returned request uses a
// graph URI under "viking://resources/graph/<sessionID>/prefers/<objectID>", a title
// of the form "User prefers <TitleizedObject>", content describing the preference for
// working memory, and LayerL1. The Metadata map populates graph-specific fields
// including subject information, relation "PREFERS", object identifiers/names, the
// session_id, and source "optimizer.heuristic".
func buildPreferenceFact(sessionID, objectID string) *contextsvc.UpsertResourceRequest {
	title := "User prefers " + titleizeFactValue(objectID)
	return &contextsvc.UpsertResourceRequest{
		URI:     "viking://resources/graph/" + sessionID + "/prefers/" + objectID,
		Title:   title,
		Content: title + " for working memory.",
		Layer:   contextsvc.LayerL1,
		Metadata: map[string]string{
			"kind":         "graph",
			"subject_id":   "user",
			"subject_type": "user",
			"subject_name": "User",
			"relation":     "PREFERS",
			"object_id":    objectID,
			"object_type":  "concept",
			"object_name":  titleizeFactValue(objectID),
			"session_id":   sessionID,
			"source":       "optimizer.heuristic",
		},
	}
}

// normalizeFactID normalizes an extracted object identifier for use in URIs.
// It lowercases the input, trims surrounding whitespace and the characters `. , ! ? : ;`, and replaces internal spaces with `-`.
func normalizeFactID(value string) string {
	trimmed := strings.TrimSpace(strings.ToLower(value))
	trimmed = strings.Trim(trimmed, " .,!?:;")
	return strings.ReplaceAll(trimmed, " ", "-")
}

// titleizeFactValue converts a normalized identifier into title-cased text.
// It replaces hyphens with spaces and capitalizes the first letter of each word, preserving the remainder of each word as-is.
func titleizeFactValue(value string) string {
	parts := strings.Fields(strings.ReplaceAll(value, "-", " "))
	for index, part := range parts {
		if part == "" {
			continue
		}
		parts[index] = strings.ToUpper(part[:1]) + part[1:]
	}
	return strings.Join(parts, " ")
}

type contextAwareProxyStream struct {
	ctx      context.Context
	requests <-chan *pb.InferenceRequest
	send     func(*pb.InferenceResponse) error
	onSend   func(*pb.InferenceResponse)
}

func (s *contextAwareProxyStream) Context() context.Context { return s.ctx }

func (s *contextAwareProxyStream) Send(resp *pb.InferenceResponse) error {
	if s.onSend != nil {
		s.onSend(resp)
	}
	return s.send(resp)
}

func (s *contextAwareProxyStream) Recv() (*pb.InferenceRequest, error) {
	select {
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	case req, ok := <-s.requests:
		if !ok {
			return nil, io.EOF
		}
		return req, nil
	}
}

func (s *contextAwareProxyStream) SetHeader(metadata.MD) error  { return nil }
func (s *contextAwareProxyStream) SendHeader(metadata.MD) error { return nil }
func (s *contextAwareProxyStream) SetTrailer(metadata.MD)       {}
func (s *contextAwareProxyStream) SendMsg(any) error            { return nil }
func (s *contextAwareProxyStream) RecvMsg(any) error            { return nil }

package runtime

import (
	"context"
	"testing"

	"github.com/ai-engine/go/internal/contextsvc"
	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// normalizeFactID
// ---------------------------------------------------------------------------

func TestNormalizeFactID(t *testing.T) {
	cases := []struct {
		input    string
		expected string
	}{
		{"dragonfly", "dragonfly"},
		{"  Dragonfly  ", "dragonfly"},
		{"Redis Cache", "redis-cache"},
		{"value.", "value"},
		{"value,", "value"},
		{"value!", "value"},
		{"value?", "value"},
		{"value:", "value"},
		{"value;", "value"},
		{"multi   word   value", "multi---word---value"},
		{"", ""},
	}
	for _, c := range cases {
		got := normalizeFactID(c.input)
		assert.Equal(t, c.expected, got, "input=%q", c.input)
	}
}

// ---------------------------------------------------------------------------
// titleizeFactValue
// ---------------------------------------------------------------------------

func TestTitleizeFactValue(t *testing.T) {
	cases := []struct {
		input    string
		expected string
	}{
		{"dragonfly", "Dragonfly"},
		{"redis-cache", "Redis Cache"},
		{"multi-word-value", "Multi Word Value"},
		{"already Title", "Already Title"},
		{"", ""},
		{"single", "Single"},
		{"one-two-three", "One Two Three"},
	}
	for _, c := range cases {
		got := titleizeFactValue(c.input)
		assert.Equal(t, c.expected, got, "input=%q", c.input)
	}
}

// ---------------------------------------------------------------------------
// cloneInferenceRequest
// ---------------------------------------------------------------------------

func TestCloneInferenceRequest_NilInput(t *testing.T) {
	got := cloneInferenceRequest(nil)
	require.NotNil(t, got)
	assert.Empty(t, got.ModelId)
	assert.Empty(t, got.Prompt)
	assert.Empty(t, got.Parameters)
	assert.Empty(t, got.ContextRefs)
}

func TestCloneInferenceRequest_DeepCopy(t *testing.T) {
	original := &pb.InferenceRequest{
		ModelId:     "model-x",
		Prompt:      "original prompt",
		Provider:    "provider-y",
		Parameters:  map[string]string{"session_id": "sess-1", "key": "val"},
		ContextRefs: []string{"ref-a", "ref-b"},
	}
	cloned := cloneInferenceRequest(original)

	// Values are equal
	assert.Equal(t, original.ModelId, cloned.ModelId)
	assert.Equal(t, original.Prompt, cloned.Prompt)
	assert.Equal(t, original.Provider, cloned.Provider)
	assert.Equal(t, original.Parameters, cloned.Parameters)
	assert.Equal(t, original.ContextRefs, cloned.ContextRefs)

	// Mutations to the clone do not affect the original
	cloned.Prompt = "mutated"
	cloned.Parameters["new_key"] = "new_val"
	cloned.ContextRefs = append(cloned.ContextRefs, "ref-c")

	assert.Equal(t, "original prompt", original.Prompt)
	assert.NotContains(t, original.Parameters, "new_key")
	assert.Len(t, original.ContextRefs, 2)
}

// ---------------------------------------------------------------------------
// formatRecentSession
// ---------------------------------------------------------------------------

func TestFormatRecentSession_NilResponse(t *testing.T) {
	got := formatRecentSession(nil, 8)
	assert.Empty(t, got)
}

func TestFormatRecentSession_EmptyEntries(t *testing.T) {
	resp := &contextsvc.SessionResponse{SessionID: "sess-1"}
	got := formatRecentSession(resp, 8)
	assert.Empty(t, got)
}

func TestFormatRecentSession_WithinLimit(t *testing.T) {
	resp := &contextsvc.SessionResponse{
		SessionID: "sess-1",
		Entries: []contextsvc.SessionEntry{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there"},
		},
	}
	got := formatRecentSession(resp, 8)
	assert.Contains(t, got, "Recent working memory:")
	assert.Contains(t, got, "- user: Hello")
	assert.Contains(t, got, "- assistant: Hi there")
}

func TestFormatRecentSession_ExceedsLimit(t *testing.T) {
	entries := make([]contextsvc.SessionEntry, 12)
	for i := range entries {
		entries[i] = contextsvc.SessionEntry{Role: "user", Content: "msg"}
	}
	// Add a distinctive last entry
	entries[11] = contextsvc.SessionEntry{Role: "assistant", Content: "final-reply"}

	resp := &contextsvc.SessionResponse{
		SessionID: "sess-1",
		Entries:   entries,
	}
	got := formatRecentSession(resp, 4)
	// Should only include the last 4 entries (entries[8..11])
	assert.Contains(t, got, "final-reply")
	// entry 0 should NOT appear (it's before the window)
	// Count lines: header + 4 content lines = 5 lines total
	lines := splitLines(got)
	assert.Equal(t, 5, len(lines))
}

func TestFormatRecentSession_LimitEqualsCount(t *testing.T) {
	resp := &contextsvc.SessionResponse{
		SessionID: "sess-1",
		Entries: []contextsvc.SessionEntry{
			{Role: "user", Content: "A"},
			{Role: "assistant", Content: "B"},
		},
	}
	got := formatRecentSession(resp, 2)
	assert.Contains(t, got, "- user: A")
	assert.Contains(t, got, "- assistant: B")
}

// ---------------------------------------------------------------------------
// formatSearchHits
// ---------------------------------------------------------------------------

func TestFormatSearchHits_NilResponse(t *testing.T) {
	got := formatSearchHits("Graph facts", nil)
	assert.Empty(t, got)
}

func TestFormatSearchHits_EmptyResults(t *testing.T) {
	resp := &contextsvc.SearchResponse{}
	got := formatSearchHits("Graph facts", resp)
	assert.Empty(t, got)
}

func TestFormatSearchHits_WithResults(t *testing.T) {
	resp := &contextsvc.SearchResponse{
		Results: []contextsvc.SearchHit{
			{ChunkText: "Dragonfly is faster than Redis."},
			{ChunkText: "Use local context for durability."},
		},
	}
	got := formatSearchHits("Graph facts", resp)
	assert.Contains(t, got, "Graph facts:")
	assert.Contains(t, got, "- Dragonfly is faster than Redis.")
	assert.Contains(t, got, "- Use local context for durability.")
}

func TestFormatSearchHits_CustomTitle(t *testing.T) {
	resp := &contextsvc.SearchResponse{
		Results: []contextsvc.SearchHit{
			{ChunkText: "Document content here."},
		},
	}
	got := formatSearchHits("Retrieved documents", resp)
	assert.Contains(t, got, "Retrieved documents:")
}

// ---------------------------------------------------------------------------
// mergeContextRefs
// ---------------------------------------------------------------------------

func TestMergeContextRefs_Empty(t *testing.T) {
	got := mergeContextRefs(nil)
	assert.Empty(t, got)
}

func TestMergeContextRefs_ExistingRefsOnly(t *testing.T) {
	got := mergeContextRefs([]string{"ref-a", "ref-b"})
	assert.Equal(t, []string{"ref-a", "ref-b"}, got)
}

func TestMergeContextRefs_DeduplicatesExisting(t *testing.T) {
	got := mergeContextRefs([]string{"ref-a", "ref-a", "ref-b"})
	assert.Equal(t, []string{"ref-a", "ref-b"}, got)
}

func TestMergeContextRefs_SkipsEmptyRefs(t *testing.T) {
	got := mergeContextRefs([]string{"ref-a", "", "ref-b"})
	assert.Equal(t, []string{"ref-a", "ref-b"}, got)
}

func TestMergeContextRefs_MergesSearchResponses(t *testing.T) {
	existing := []string{"ref-a"}
	resp := &contextsvc.SearchResponse{
		Results: []contextsvc.SearchHit{
			{URI: "viking://graph/x"},
			{URI: "viking://docs/y"},
		},
	}
	got := mergeContextRefs(existing, resp)
	assert.Equal(t, []string{"ref-a", "viking://graph/x", "viking://docs/y"}, got)
}

func TestMergeContextRefs_DeduplicatesAcrossSources(t *testing.T) {
	existing := []string{"viking://graph/x"}
	resp := &contextsvc.SearchResponse{
		Results: []contextsvc.SearchHit{
			{URI: "viking://graph/x"}, // duplicate
			{URI: "viking://docs/y"}, // new
		},
	}
	got := mergeContextRefs(existing, resp)
	assert.Equal(t, []string{"viking://graph/x", "viking://docs/y"}, got)
}

func TestMergeContextRefs_SkipsEmptyURIsInResponses(t *testing.T) {
	resp := &contextsvc.SearchResponse{
		Results: []contextsvc.SearchHit{
			{URI: ""},
			{URI: "viking://docs/y"},
		},
	}
	got := mergeContextRefs(nil, resp)
	assert.Equal(t, []string{"viking://docs/y"}, got)
}

func TestMergeContextRefs_NilResponseSkipped(t *testing.T) {
	got := mergeContextRefs([]string{"ref-a"}, nil)
	assert.Equal(t, []string{"ref-a"}, got)
}

// ---------------------------------------------------------------------------
// learnFromTurn (via ContextAwareService.learnFromTurn)
// ---------------------------------------------------------------------------

func TestLearnFromTurn_NoOpWhenContextIsNil(t *testing.T) {
	svc := &ContextAwareService{context: nil}
	// Must not panic
	svc.learnFromTurn(context.Background(), "sess-1", "prefer dragonfly over redis", "use dragonfly")
}

func TestLearnFromTurn_NoOpWhenSessionIDEmpty(t *testing.T) {
	backend := &contextBackendStub{enabled: true}
	svc := NewContextAwareService(&runtimeRecorder{}, backend)
	svc.learnFromTurn(context.Background(), "", "prefer dragonfly over redis", "yes")
	assert.Empty(t, backend.upsertRequests)
}

func TestLearnFromTurn_NoOpWhenReplyEmpty(t *testing.T) {
	backend := &contextBackendStub{enabled: true}
	svc := NewContextAwareService(&runtimeRecorder{}, backend)
	svc.learnFromTurn(context.Background(), "sess-1", "prefer dragonfly", "")
	assert.Empty(t, backend.upsertRequests)
}

func TestLearnFromTurn_PreferOverPattern(t *testing.T) {
	backend := &contextBackendStub{enabled: true}
	svc := NewContextAwareService(&runtimeRecorder{}, backend)
	svc.learnFromTurn(context.Background(), "sess-x", "I prefer Redis over Dragonfly", "Understood.")
	require.Len(t, backend.upsertRequests, 1)
	req := backend.upsertRequests[0]
	assert.Equal(t, "redis", req.Metadata["object_id"])
	assert.Equal(t, "PREFERS", req.Metadata["relation"])
	assert.Equal(t, contextsvc.LayerL1, req.Layer)
}

func TestLearnFromTurn_DragonflyPlusPreferKeyword(t *testing.T) {
	backend := &contextBackendStub{enabled: true}
	svc := NewContextAwareService(&runtimeRecorder{}, backend)
	// No "over" pattern but contains "dragonfly" and "prefer"
	svc.learnFromTurn(context.Background(), "sess-y", "we prefer dragonfly", "sure")
	require.Len(t, backend.upsertRequests, 1)
	assert.Equal(t, "dragonfly", backend.upsertRequests[0].Metadata["object_id"])
}

func TestLearnFromTurn_DragonflyPlusWorkingMemory(t *testing.T) {
	backend := &contextBackendStub{enabled: true}
	svc := NewContextAwareService(&runtimeRecorder{}, backend)
	svc.learnFromTurn(context.Background(), "sess-z", "dragonfly working memory", "yes")
	require.Len(t, backend.upsertRequests, 1)
	assert.Equal(t, "dragonfly", backend.upsertRequests[0].Metadata["object_id"])
}

func TestLearnFromTurn_NoPatternMatchDoesNotUpsert(t *testing.T) {
	backend := &contextBackendStub{enabled: true}
	svc := NewContextAwareService(&runtimeRecorder{}, backend)
	svc.learnFromTurn(context.Background(), "sess-n", "tell me about redis", "redis is fast")
	assert.Empty(t, backend.upsertRequests)
}

// ---------------------------------------------------------------------------
// ContextAwareService — disabled context path
// ---------------------------------------------------------------------------

func TestContextAwareServicePassesThroughWhenContextDisabled(t *testing.T) {
	inner := &runtimeRecorder{}
	backend := &contextBackendStub{enabled: false}
	svc := NewContextAwareService(inner, backend)

	stream := &inferenceStreamHarness{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{
				ModelId: "model-x",
				Prompt:  "Direct question.",
				Parameters: map[string]string{
					"session_id": "sess-disabled",
				},
			},
		},
	}

	err := svc.StreamInference(stream.ctx, stream)
	require.NoError(t, err)
	// The inner recorder should have received the original unmodified request
	require.Len(t, inner.seen, 1)
	assert.Equal(t, "Direct question.", inner.seen[0].GetPrompt())
	// No context operations should have been performed
	assert.Empty(t, backend.appendRequests)
	assert.Empty(t, backend.searchRequests)
}

func TestContextAwareServicePassesThroughWhenContextIsNil(t *testing.T) {
	inner := &runtimeRecorder{}
	svc := &ContextAwareService{
		inner:   inner,
		context: nil,
	}

	stream := &inferenceStreamHarness{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{Prompt: "Hello."},
		},
	}

	err := svc.StreamInference(stream.ctx, stream)
	require.NoError(t, err)
	require.Len(t, inner.seen, 1)
}

// ---------------------------------------------------------------------------
// ContextAwareService — no session_id does not assemble context
// ---------------------------------------------------------------------------

func TestContextAwareServiceNoSessionIDSkipsContextAssembly(t *testing.T) {
	inner := &runtimeRecorder{}
	backend := &contextBackendStub{
		enabled: true,
		session: &contextsvc.SessionResponse{
			SessionID: "sess-1",
			Entries: []contextsvc.SessionEntry{
				{Role: "user", Content: "old message"},
			},
		},
	}
	svc := NewContextAwareService(inner, backend)

	stream := &inferenceStreamHarness{
		ctx: context.Background(),
		requests: []*pb.InferenceRequest{
			{
				ModelId:    "model-x",
				Prompt:     "No session.",
				Parameters: map[string]string{}, // no session_id
			},
		},
	}

	err := svc.StreamInference(stream.ctx, stream)
	require.NoError(t, err)
	require.Len(t, inner.seen, 1)
	// Prompt should be unchanged (no assembly)
	assert.Equal(t, "No session.", inner.seen[0].GetPrompt())
	// No search or append calls should have occurred
	assert.Empty(t, backend.searchRequests)
	assert.Empty(t, backend.appendRequests)
}

// ---------------------------------------------------------------------------
// ContextAwareService — delegated non-inference methods
// ---------------------------------------------------------------------------

func TestContextAwareServiceDelegatesGetStatus(t *testing.T) {
	inner := &runtimeRecorder{}
	svc := NewContextAwareService(inner, &contextBackendStub{enabled: true})
	_, err := svc.GetStatus(context.Background(), nil)
	require.NoError(t, err)
}

func TestContextAwareServiceDelegatesListModels(t *testing.T) {
	inner := &runtimeRecorder{}
	svc := NewContextAwareService(inner, &contextBackendStub{enabled: true})
	_, err := svc.ListModels(context.Background(), nil)
	require.NoError(t, err)
}

func TestContextAwareServiceLoadedModelCountDelegates(t *testing.T) {
	inner := &runtimeRecorder{}
	svc := NewContextAwareService(inner, &contextBackendStub{enabled: true})
	assert.Equal(t, 0, svc.LoadedModelCount())
}

// ---------------------------------------------------------------------------
// buildPreferenceFact
// ---------------------------------------------------------------------------

func TestBuildPreferenceFact(t *testing.T) {
	fact := buildPreferenceFact("sess-1", "dragonfly")
	require.NotNil(t, fact)
	assert.Equal(t, "viking://resources/graph/sess-1/prefers/dragonfly", fact.URI)
	assert.Equal(t, "User prefers Dragonfly", fact.Title)
	assert.Contains(t, fact.Content, "User prefers Dragonfly")
	assert.Equal(t, contextsvc.LayerL1, fact.Layer)
	assert.Equal(t, "graph", fact.Metadata["kind"])
	assert.Equal(t, "PREFERS", fact.Metadata["relation"])
	assert.Equal(t, "dragonfly", fact.Metadata["object_id"])
	assert.Equal(t, "Dragonfly", fact.Metadata["object_name"])
	assert.Equal(t, "sess-1", fact.Metadata["session_id"])
	assert.Equal(t, "optimizer.heuristic", fact.Metadata["source"])
}

func TestBuildPreferenceFact_MultiWordObjectID(t *testing.T) {
	fact := buildPreferenceFact("sess-2", "redis-cache")
	assert.Equal(t, "viking://resources/graph/sess-2/prefers/redis-cache", fact.URI)
	assert.Equal(t, "User prefers Redis Cache", fact.Title)
	assert.Equal(t, "Redis Cache", fact.Metadata["object_name"])
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}
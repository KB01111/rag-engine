package contextsvc

import (
	"testing"
	"time"

	pb "github.com/ai-engine/proto/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// ---------------------------------------------------------------------------
// layerToProto
// ---------------------------------------------------------------------------

func TestLayerToProto_AllBranches(t *testing.T) {
	cases := []struct {
		layer    Layer
		expected pb.ContextLayer
	}{
		{LayerL0, pb.ContextLayer_CONTEXT_LAYER_L0},
		{LayerL1, pb.ContextLayer_CONTEXT_LAYER_L1},
		{LayerL2, pb.ContextLayer_CONTEXT_LAYER_L2},
		{"unknown", pb.ContextLayer_CONTEXT_LAYER_UNSPECIFIED},
		{"", pb.ContextLayer_CONTEXT_LAYER_UNSPECIFIED},
		{"L99", pb.ContextLayer_CONTEXT_LAYER_UNSPECIFIED},
	}
	for _, c := range cases {
		got := layerToProto(c.layer)
		assert.Equal(t, c.expected, got, "layer %q", c.layer)
	}
}

// ---------------------------------------------------------------------------
// statusFromProto
// ---------------------------------------------------------------------------

func TestStatusFromProto_NilInput(t *testing.T) {
	got := statusFromProto(nil)
	require.NotNil(t, got)
	assert.Equal(t, int64(0), got.DocumentCount)
	assert.Equal(t, int64(0), got.ChunkCount)
	assert.False(t, got.Ready)
	assert.Empty(t, got.ManagedRoots)
}

func TestStatusFromProto_ValidInput(t *testing.T) {
	proto := &pb.ContextStatus{
		DocumentCount:  42,
		ChunkCount:     100,
		IndexSizeBytes: 1024,
		EmbeddingModel: "all-minilm-l6",
		Ready:          true,
		ManagedRoots:   []string{"workspace", "docs"},
	}
	got := statusFromProto(proto)
	assert.Equal(t, int64(42), got.DocumentCount)
	assert.Equal(t, int64(100), got.ChunkCount)
	assert.Equal(t, int64(1024), got.IndexSizeBytes)
	assert.Equal(t, "all-minilm-l6", got.EmbeddingModel)
	assert.True(t, got.Ready)
	assert.Equal(t, []string{"workspace", "docs"}, got.ManagedRoots)
}

// ---------------------------------------------------------------------------
// listResourcesFromProto
// ---------------------------------------------------------------------------

func TestListResourcesFromProto_NilInput(t *testing.T) {
	got := listResourcesFromProto(nil)
	require.NotNil(t, got)
	assert.Nil(t, got.Resources)
}

func TestListResourcesFromProto_EmptyList(t *testing.T) {
	got := listResourcesFromProto(&pb.ContextResourceList{})
	require.NotNil(t, got)
	assert.Empty(t, got.Resources)
}

func TestListResourcesFromProto_PopulatedList(t *testing.T) {
	proto := &pb.ContextResourceList{
		Resources: []*pb.ContextResource{
			{
				Uri:      "viking://resources/graph/project-x",
				Title:    "Project X",
				Layer:    "l1",
				Metadata: map[string]string{"kind": "graph"},
			},
			{
				Uri:   "viking://resources/docs/readme.md",
				Title: "Readme",
				Layer: "l2",
			},
		},
	}
	got := listResourcesFromProto(proto)
	require.Len(t, got.Resources, 2)
	assert.Equal(t, "viking://resources/graph/project-x", got.Resources[0].URI)
	assert.Equal(t, "Project X", got.Resources[0].Title)
	assert.Equal(t, Layer("l1"), got.Resources[0].Layer)
	assert.Equal(t, map[string]string{"kind": "graph"}, got.Resources[0].Metadata)
	assert.Equal(t, "viking://resources/docs/readme.md", got.Resources[1].URI)
}

// ---------------------------------------------------------------------------
// upsertResponseFromProto
// ---------------------------------------------------------------------------

func TestUpsertResponseFromProto_NilInput(t *testing.T) {
	got := upsertResponseFromProto(nil)
	require.NotNil(t, got)
	assert.Equal(t, int32(0), got.ChunksIndexed)
	assert.Empty(t, got.Resource.URI)
}

func TestUpsertResponseFromProto_NilResource(t *testing.T) {
	proto := &pb.ContextUpsertResourceResponse{
		Resource:      nil,
		ChunksIndexed: 5,
	}
	got := upsertResponseFromProto(proto)
	assert.Equal(t, int32(5), got.ChunksIndexed)
	assert.Empty(t, got.Resource.URI)
}

func TestUpsertResponseFromProto_ValidInput(t *testing.T) {
	proto := &pb.ContextUpsertResourceResponse{
		Resource: &pb.ContextResource{
			Uri:      "viking://resources/graph/project-x",
			Title:    "Project X",
			Layer:    "l2",
			Metadata: map[string]string{"source": "test"},
		},
		ChunksIndexed: 3,
	}
	got := upsertResponseFromProto(proto)
	assert.Equal(t, "viking://resources/graph/project-x", got.Resource.URI)
	assert.Equal(t, "Project X", got.Resource.Title)
	assert.Equal(t, Layer("l2"), got.Resource.Layer)
	assert.Equal(t, int32(3), got.ChunksIndexed)
}

// ---------------------------------------------------------------------------
// searchResponseFromProto
// ---------------------------------------------------------------------------

func TestSearchResponseFromProto_NilInput(t *testing.T) {
	got := searchResponseFromProto(nil)
	require.NotNil(t, got)
	assert.Nil(t, got.Results)
	assert.Equal(t, float64(0), got.QueryTimeMs)
}

func TestSearchResponseFromProto_EmptyResults(t *testing.T) {
	got := searchResponseFromProto(&pb.ContextSearchResponse{QueryTimeMs: 1.5})
	require.NotNil(t, got)
	assert.Empty(t, got.Results)
	assert.Equal(t, 1.5, got.QueryTimeMs)
}

func TestSearchResponseFromProto_PopulatedResults(t *testing.T) {
	proto := &pb.ContextSearchResponse{
		Results: []*pb.ContextSearchResult{
			{
				Uri:        "viking://resources/graph/project-x",
				DocumentId: "project-x",
				ChunkText:  "Project X uses Dragonfly.",
				Score:      0.95,
				Metadata:   map[string]string{"kind": "graph"},
				Layer:      "l1",
			},
		},
		QueryTimeMs: 2.3,
	}
	got := searchResponseFromProto(proto)
	require.Len(t, got.Results, 1)
	assert.Equal(t, "viking://resources/graph/project-x", got.Results[0].URI)
	assert.Equal(t, "project-x", got.Results[0].DocumentID)
	assert.Equal(t, "Project X uses Dragonfly.", got.Results[0].ChunkText)
	assert.InDelta(t, 0.95, got.Results[0].Score, 0.001)
	assert.Equal(t, Layer("l1"), got.Results[0].Layer)
	assert.Equal(t, 2.3, got.QueryTimeMs)
}

// ---------------------------------------------------------------------------
// workspaceSyncFromProto
// ---------------------------------------------------------------------------

func TestWorkspaceSyncFromProto_NilInput(t *testing.T) {
	got := workspaceSyncFromProto(nil)
	require.NotNil(t, got)
	assert.Empty(t, got.Root)
	assert.Nil(t, got.Prefix)
}

func TestWorkspaceSyncFromProto_ValidInput(t *testing.T) {
	prefix := "workspace/"
	proto := &pb.ContextWorkspaceSyncResponse{
		Root:               "workspace",
		Prefix:             &prefix,
		IndexedResources:   10,
		ReindexedResources: 2,
		DeletedResources:   1,
		SkippedFiles:       3,
	}
	got := workspaceSyncFromProto(proto)
	assert.Equal(t, "workspace", got.Root)
	require.NotNil(t, got.Prefix)
	assert.Equal(t, "workspace/", *got.Prefix)
	assert.Equal(t, int32(10), got.IndexedResources)
	assert.Equal(t, int32(2), got.ReindexedResources)
	assert.Equal(t, int32(1), got.DeletedResources)
	assert.Equal(t, int32(3), got.SkippedFiles)
}

func TestWorkspaceSyncFromProto_NilPrefix(t *testing.T) {
	proto := &pb.ContextWorkspaceSyncResponse{
		Root:   "workspace",
		Prefix: nil,
	}
	got := workspaceSyncFromProto(proto)
	assert.Equal(t, "workspace", got.Root)
	assert.Nil(t, got.Prefix)
}

// ---------------------------------------------------------------------------
// fileListFromProto
// ---------------------------------------------------------------------------

func TestFileListFromProto_NilInput(t *testing.T) {
	got := fileListFromProto(nil)
	require.NotNil(t, got)
	assert.Nil(t, got.Entries)
}

func TestFileListFromProto_EmptyEntries(t *testing.T) {
	got := fileListFromProto(&pb.ContextFileListResponse{})
	assert.Empty(t, got.Entries)
}

func TestFileListFromProto_EntriesWithAndWithoutVersion(t *testing.T) {
	ver := int64(7)
	proto := &pb.ContextFileListResponse{
		Entries: []*pb.ContextFileEntry{
			{
				Name:      "main.go",
				Path:      "src/main.go",
				IsDir:     false,
				SizeBytes: 1024,
				Version:   &ver,
			},
			{
				Name:  "docs",
				Path:  "docs/",
				IsDir: true,
			},
		},
	}
	got := fileListFromProto(proto)
	require.Len(t, got.Entries, 2)

	// Entry with version set
	assert.Equal(t, "main.go", got.Entries[0].Name)
	assert.Equal(t, "src/main.go", got.Entries[0].Path)
	assert.False(t, got.Entries[0].IsDir)
	assert.Equal(t, int64(1024), got.Entries[0].SizeBytes)
	require.NotNil(t, got.Entries[0].Version)
	assert.Equal(t, int64(7), *got.Entries[0].Version)

	// Entry without version (nil)
	assert.Equal(t, "docs", got.Entries[1].Name)
	assert.True(t, got.Entries[1].IsDir)
	assert.Nil(t, got.Entries[1].Version)
}

// ---------------------------------------------------------------------------
// sessionResponseFromProto
// ---------------------------------------------------------------------------

func TestSessionResponseFromProto_NilInput(t *testing.T) {
	got := sessionResponseFromProto(nil)
	require.NotNil(t, got)
	assert.Empty(t, got.SessionID)
	assert.Nil(t, got.Entries)
}

func TestSessionResponseFromProto_EmptyEntries(t *testing.T) {
	got := sessionResponseFromProto(&pb.ContextSessionHistory{SessionId: "sess-1"})
	assert.Equal(t, "sess-1", got.SessionID)
	assert.Empty(t, got.Entries)
}

func TestSessionResponseFromProto_ValidEntries(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Millisecond)
	proto := &pb.ContextSessionHistory{
		SessionId: "sess-42",
		Entries: []*pb.ContextSessionEntry{
			{
				SessionId: "sess-42",
				Role:      "user",
				Content:   "Project X uses Dragonfly.",
				Metadata:  map[string]string{"source": "test"},
				CreatedAt: timestamppb.New(now),
			},
			{
				SessionId: "sess-42",
				Role:      "assistant",
				Content:   "Dragonfly is the preferred store.",
				CreatedAt: nil,
			},
		},
	}
	got := sessionResponseFromProto(proto)
	assert.Equal(t, "sess-42", got.SessionID)
	require.Len(t, got.Entries, 2)

	// First entry has timestamp
	assert.Equal(t, "user", got.Entries[0].Role)
	assert.Equal(t, "Project X uses Dragonfly.", got.Entries[0].Content)
	assert.Equal(t, now.UnixMilli(), got.Entries[0].CreatedAt)
	assert.Equal(t, map[string]string{"source": "test"}, got.Entries[0].Metadata)

	// Second entry has nil timestamp → CreatedAt should be 0
	assert.Equal(t, "assistant", got.Entries[1].Role)
	assert.Equal(t, int64(0), got.Entries[1].CreatedAt)
}

func TestSessionResponseFromProto_ZeroTimestampIsZero(t *testing.T) {
	// A nil CreatedAt pointer must yield CreatedAt == 0 in the response
	proto := &pb.ContextSessionHistory{
		SessionId: "sess-ts",
		Entries: []*pb.ContextSessionEntry{
			{
				SessionId: "sess-ts",
				Role:      "user",
				Content:   "Hello",
				CreatedAt: nil,
			},
		},
	}
	got := sessionResponseFromProto(proto)
	require.Len(t, got.Entries, 1)
	assert.Equal(t, int64(0), got.Entries[0].CreatedAt)
}
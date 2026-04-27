package daemon

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	pb "github.com/ai-engine/proto/go"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"
)

const defaultCountCacheTTL = 2 * time.Second

type Client struct {
	conn     *grpc.ClientConn
	runtime  pb.RuntimeClient
	rag      pb.RagClient
	context  pb.ContextClient
	training pb.TrainingClient
	mcp      pb.MCPClient

	mcpMu          sync.RWMutex
	mcpConnections map[string]struct{}

	counts        countCache
	countCacheTTL time.Duration
}

type countCache struct {
	mu                   sync.Mutex
	loadedModelCount     int
	loadedModelExpires   time.Time
	documentCount        int64
	documentCountExpires time.Time
	activeRunCount       int
	activeRunExpires     time.Time
}

// NewClient creates and returns a Client connected to the daemon at addr.
// It wraps the provided context with a 10-second timeout, validates the daemon
// address (rejecting non-loopback addresses when TLS would be required), and
// dials the daemon. On success it constructs a Client with gRPC service
// clients and an empty MCP connection bookkeeping map. Returns an error if
// NewClient creates a Client connected to the daemon at addr.
// It wraps ctx with a 10-second timeout, validates addr to ensure an insecure connection is only used for allowed loopback addresses, dials the daemon (blocking) using insecure credentials, and returns a Client with all service stubs initialized. An error is returned if address validation or dialing fails.
func NewClient(ctx context.Context, addr string) (*Client, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Validate address and ensure insecure credentials are only used for loopback
	if err := validateDaemonAddress(addr); err != nil {
		return nil, fmt.Errorf("daemon address validation failed: %w", err)
	}

	conn, err := grpc.DialContext(
		ctx,
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(64<<20),
			grpc.MaxCallSendMsgSize(64<<20),
		),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		return nil, err
	}

	return &Client{
		conn:           conn,
		runtime:        pb.NewRuntimeClient(conn),
		rag:            pb.NewRagClient(conn),
		context:        pb.NewContextClient(conn),
		training:       pb.NewTrainingClient(conn),
		mcp:            pb.NewMCPClient(conn),
		mcpConnections: make(map[string]struct{}),
		countCacheTTL:  defaultCountCacheTTL,
	}, nil
}

func validateDaemonAddress(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("invalid address format: %w", err)
	}

	// Allow empty host (defaults to localhost) or explicit loopback
	if host == "" || host == "localhost" || host == "127.0.0.1" || host == "::1" {
		return nil
	}

	// Parse the IP to check if it's loopback
	ip := net.ParseIP(host)
	if ip != nil && ip.IsLoopback() {
		return nil
	}

	// Non-loopback addresses require TLS (not yet implemented)
	return fmt.Errorf("non-loopback daemon address %q requires TLS credentials (not yet supported)", host)
}

func (c *Client) Close() error {
	return c.conn.Close()
}

func (c *Client) GetStatus(ctx context.Context, req *emptypb.Empty) (*pb.RuntimeStatus, error) {
	return c.runtime.GetStatus(withOutgoingRequestID(ctx), req)
}

func (c *Client) ListModels(ctx context.Context, req *emptypb.Empty) (*pb.ModelList, error) {
	return c.runtime.ListModels(withOutgoingRequestID(ctx), req)
}

func (c *Client) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.ModelInfo, error) {
	return c.runtime.LoadModel(withOutgoingRequestID(ctx), req)
}

func (c *Client) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*emptypb.Empty, error) {
	return c.runtime.UnloadModel(withOutgoingRequestID(ctx), req)
}

func (c *Client) StreamInference(ctx context.Context, stream pb.Runtime_StreamInferenceServer) error {
	ctx = withOutgoingRequestID(ctx)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	group, groupCtx := errgroup.WithContext(ctx)

	clientStream, err := c.runtime.StreamInference(groupCtx)
	if err != nil {
		return err
	}
	defer func() {
		cancel()
		_ = clientStream.CloseSend()
	}()

	group.Go(func() error {
		for {
			select {
			case <-groupCtx.Done():
				return groupCtx.Err()
			default:
			}

			req, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				return clientStream.CloseSend()
			}
			if err != nil {
				cancel()
				return err
			}
			if err := clientStream.Send(req); err != nil {
				cancel()
				return err
			}
		}
	})
	group.Go(func() error {
		for {
			select {
			case <-groupCtx.Done():
				return groupCtx.Err()
			default:
			}

			resp, err := clientStream.Recv()
			if errors.Is(err, io.EOF) {
				return nil
			}
			if err != nil {
				cancel()
				return err
			}
			if err := stream.Send(resp); err != nil {
				cancel()
				return err
			}
		}
	})

	return group.Wait()
}

func (c *Client) LoadedModelCount() int {
	if count, ok := c.cachedLoadedModelCount(); ok {
		return count
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	status, err := c.runtime.GetStatus(ctx, &emptypb.Empty{})
	if err != nil {
		return 0
	}
	count := len(status.LoadedModels)
	c.storeLoadedModelCount(count)
	return count
}

func (c *Client) UpsertDocument(ctx context.Context, req *pb.UpsertRequest) (*pb.UpsertResponse, error) {
	return c.rag.UpsertDocument(withOutgoingRequestID(ctx), req)
}

func (c *Client) DeleteDocument(ctx context.Context, req *pb.DeleteRequest) (*emptypb.Empty, error) {
	return c.rag.DeleteDocument(withOutgoingRequestID(ctx), req)
}

func (c *Client) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	return c.rag.Search(withOutgoingRequestID(ctx), req)
}

func (c *Client) GetRagStatus(ctx context.Context, req *emptypb.Empty) (*pb.RagStatus, error) {
	return c.rag.GetRagStatus(withOutgoingRequestID(ctx), req)
}

func (c *Client) ListDocuments(ctx context.Context, req *emptypb.Empty) (*pb.DocumentList, error) {
	return c.rag.ListDocuments(withOutgoingRequestID(ctx), req)
}

func (c *Client) DocumentCount() int64 {
	if count, ok := c.cachedDocumentCount(); ok {
		return count
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	status, err := c.rag.GetRagStatus(ctx, &emptypb.Empty{})
	if err != nil {
		return 0
	}
	c.storeDocumentCount(status.DocumentCount)
	return status.DocumentCount
}

func (c *Client) GetContextStatus(ctx context.Context, req *emptypb.Empty) (*pb.ContextStatus, error) {
	return c.context.GetContextStatus(withOutgoingRequestID(ctx), req)
}

func (c *Client) ListResources(ctx context.Context, req *emptypb.Empty) (*pb.ContextResourceList, error) {
	return c.context.ListResources(withOutgoingRequestID(ctx), req)
}

func (c *Client) UpsertResource(ctx context.Context, req *pb.ContextUpsertResourceRequest) (*pb.ContextUpsertResourceResponse, error) {
	return c.context.UpsertResource(withOutgoingRequestID(ctx), req)
}

func (c *Client) DeleteResource(ctx context.Context, req *pb.ContextDeleteResourceRequest) (*emptypb.Empty, error) {
	return c.context.DeleteResource(withOutgoingRequestID(ctx), req)
}

func (c *Client) SearchContext(ctx context.Context, req *pb.ContextSearchRequest) (*pb.ContextSearchResponse, error) {
	return c.context.SearchContext(withOutgoingRequestID(ctx), req)
}

func (c *Client) SyncWorkspace(ctx context.Context, req *pb.ContextWorkspaceSyncRequest) (*pb.ContextWorkspaceSyncResponse, error) {
	return c.context.SyncWorkspace(withOutgoingRequestID(ctx), req)
}

func (c *Client) ListFiles(ctx context.Context, req *pb.ContextFileListRequest) (*pb.ContextFileListResponse, error) {
	return c.context.ListFiles(withOutgoingRequestID(ctx), req)
}

func (c *Client) ReadFile(ctx context.Context, req *pb.ContextFileReadRequest) (*pb.ContextFileReadResponse, error) {
	return c.context.ReadFile(withOutgoingRequestID(ctx), req)
}

func (c *Client) WriteFile(ctx context.Context, req *pb.ContextFileWriteRequest) (*pb.ContextFileWriteResponse, error) {
	return c.context.WriteFile(withOutgoingRequestID(ctx), req)
}

func (c *Client) DeleteFile(ctx context.Context, req *pb.ContextFileDeleteRequest) (*pb.ContextFileDeleteResponse, error) {
	return c.context.DeleteFile(withOutgoingRequestID(ctx), req)
}

func (c *Client) MoveFile(ctx context.Context, req *pb.ContextFileMoveRequest) (*pb.ContextFileMoveResponse, error) {
	return c.context.MoveFile(withOutgoingRequestID(ctx), req)
}

func (c *Client) AppendSession(ctx context.Context, req *pb.ContextSessionAppendRequest) (*pb.ContextSessionHistory, error) {
	return c.context.AppendSession(withOutgoingRequestID(ctx), req)
}

func (c *Client) GetSession(ctx context.Context, req *pb.ContextSessionGetRequest) (*pb.ContextSessionHistory, error) {
	return c.context.GetSession(withOutgoingRequestID(ctx), req)
}

func (c *Client) StartRun(ctx context.Context, req *pb.TrainingRunRequest) (*pb.TrainingRun, error) {
	return c.training.StartRun(withOutgoingRequestID(ctx), req)
}

func (c *Client) CancelRun(ctx context.Context, req *pb.CancelRequest) (*emptypb.Empty, error) {
	return c.training.CancelRun(withOutgoingRequestID(ctx), req)
}

func (c *Client) ListRuns(ctx context.Context, req *emptypb.Empty) (*pb.TrainingRunList, error) {
	return c.training.ListRuns(withOutgoingRequestID(ctx), req)
}

func (c *Client) ListArtifacts(ctx context.Context, req *pb.ArtifactsRequest) (*pb.ArtifactList, error) {
	return c.training.ListArtifacts(withOutgoingRequestID(ctx), req)
}

func (c *Client) StreamLogs(req *pb.LogsRequest, stream pb.Training_StreamLogsServer) error {
	clientStream, err := c.training.StreamLogs(withOutgoingRequestID(stream.Context()), req)
	if err != nil {
		return err
	}

	for {
		entry, err := clientStream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := stream.Send(entry); err != nil {
			return err
		}
	}
}

func (c *Client) ActiveRunCount() int {
	if count, ok := c.cachedActiveRunCount(); ok {
		return count
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	runs, err := c.training.ListRuns(ctx, &emptypb.Empty{})
	if err != nil {
		return 0
	}

	count := 0
	for _, run := range runs.Runs {
		if run.Status == "queued" || run.Status == "starting" || run.Status == "running" {
			count++
		}
	}
	c.storeActiveRunCount(count)
	return count
}

func (c *Client) Connect(ctx context.Context, req *pb.MCPConnectionRequest) (*pb.MCPConnection, error) {
	connection, err := c.mcp.Connect(withOutgoingRequestID(ctx), req)
	if err != nil {
		return nil, err
	}
	c.mcpMu.Lock()
	c.mcpConnections[connection.ConnectionId] = struct{}{}
	c.mcpMu.Unlock()
	return connection, nil
}

func (c *Client) Disconnect(ctx context.Context, req *pb.DisconnectRequest) (*emptypb.Empty, error) {
	response, err := c.mcp.Disconnect(withOutgoingRequestID(ctx), req)
	if err != nil {
		return nil, err
	}
	c.mcpMu.Lock()
	delete(c.mcpConnections, req.ConnectionId)
	c.mcpMu.Unlock()
	return response, nil
}

func (c *Client) ListTools(ctx context.Context, req *pb.MCPConnectionRequest) (*pb.ToolList, error) {
	return c.mcp.ListTools(withOutgoingRequestID(ctx), req)
}

func (c *Client) CallTool(ctx context.Context, req *pb.CallToolRequest) (*pb.CallToolResponse, error) {
	return c.mcp.CallTool(withOutgoingRequestID(ctx), req)
}

func (c *Client) ConnectionCount() int {
	c.mcpMu.RLock()
	defer c.mcpMu.RUnlock()
	return len(c.mcpConnections)
}

func (c *Client) cacheTTL() time.Duration {
	if c.countCacheTTL > 0 {
		return c.countCacheTTL
	}
	return defaultCountCacheTTL
}

func (c *Client) cachedLoadedModelCount() (int, bool) {
	c.counts.mu.Lock()
	defer c.counts.mu.Unlock()
	if time.Now().Before(c.counts.loadedModelExpires) {
		return c.counts.loadedModelCount, true
	}
	return 0, false
}

func (c *Client) storeLoadedModelCount(count int) {
	c.counts.mu.Lock()
	defer c.counts.mu.Unlock()
	c.counts.loadedModelCount = count
	c.counts.loadedModelExpires = time.Now().Add(c.cacheTTL())
}

func (c *Client) cachedDocumentCount() (int64, bool) {
	c.counts.mu.Lock()
	defer c.counts.mu.Unlock()
	if time.Now().Before(c.counts.documentCountExpires) {
		return c.counts.documentCount, true
	}
	return 0, false
}

func (c *Client) storeDocumentCount(count int64) {
	c.counts.mu.Lock()
	defer c.counts.mu.Unlock()
	c.counts.documentCount = count
	c.counts.documentCountExpires = time.Now().Add(c.cacheTTL())
}

func (c *Client) cachedActiveRunCount() (int, bool) {
	c.counts.mu.Lock()
	defer c.counts.mu.Unlock()
	if time.Now().Before(c.counts.activeRunExpires) {
		return c.counts.activeRunCount, true
	}
	return 0, false
}

func (c *Client) storeActiveRunCount(count int) {
	c.counts.mu.Lock()
	defer c.counts.mu.Unlock()
	c.counts.activeRunCount = count
	c.counts.activeRunExpires = time.Now().Add(c.cacheTTL())
}

func withOutgoingRequestID(ctx context.Context) context.Context {
	if md, ok := metadata.FromOutgoingContext(ctx); ok && len(md.Get("x-request-id")) > 0 {
		return ctx
	}
	id := ""
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if values := md.Get("x-request-id"); len(values) > 0 {
			id = values[0]
		}
	}
	if id == "" {
		return ctx
	}
	return metadata.AppendToOutgoingContext(ctx, "x-request-id", id)
}

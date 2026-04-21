package mcp

import (
	"context"
	"fmt"
	"sync"

	"github.com/ai-engine/go/internal/config"
	pb "github.com/ai-engine/proto/go"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/structpb"
)

type Manager struct {
	mu          sync.RWMutex
	connections map[string]*Connection
	config      *config.Config
}

type Connection struct {
	ID         string
	ServerURL  string
	ServerName string
	Connected  bool
	Auth       map[string]string
	Tools      map[string]*pb.Tool
}

func NewManager(cfg *config.Config) *Manager {
	return &Manager{
		connections: make(map[string]*Connection),
		config:      cfg,
	}
}

func (m *Manager) Connect(ctx context.Context, req *pb.MCPConnectionRequest) (*pb.MCPConnection, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	connID := fmt.Sprintf("conn-%s", req.ServerUrl)

	// In production, this would establish actual MCP connection
	// For now, simulate connection
	conn := &Connection{
		ID:         connID,
		ServerURL:  req.ServerUrl,
		ServerName: "MCP Server",
		Connected:  true,
		Auth:       req.Auth,
		Tools:      make(map[string]*pb.Tool),
	}

	// Simulate some available tools
	conn.Tools["get_weather"] = &pb.Tool{
		Name:        "get_weather",
		Description: "Get current weather for a location",
		Parameters: []*pb.ToolParameter{
			{Name: "location", Type: "string", Required: true, Description: "City name"},
		},
	}
	conn.Tools["search_files"] = &pb.Tool{
		Name:        "search_files",
		Description: "Search files in a directory",
		Parameters: []*pb.ToolParameter{
			{Name: "query", Type: "string", Required: true, Description: "Search query"},
			{Name: "path", Type: "string", Required: false, Description: "Directory path"},
		},
	}

	m.connections[connID] = conn

	return &pb.MCPConnection{
		ConnectionId: connID,
		Connected:    conn.Connected,
		ServerName:   conn.ServerName,
	}, nil
}

func (m *Manager) Disconnect(ctx context.Context, req *pb.DisconnectRequest) (*emptypb.Empty, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	conn, exists := m.connections[req.ConnectionId]
	if !exists {
		return nil, fmt.Errorf("connection not found: %s", req.ConnectionId)
	}

	conn.Connected = false
	delete(m.connections, req.ConnectionId)

	return &emptypb.Empty{}, nil
}

func (m *Manager) ListTools(ctx context.Context, req *pb.MCPConnectionRequest) (*pb.ToolList, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Find connection by URL (simplified)
	var conn *Connection
	for _, c := range m.connections {
		if c.ServerURL == req.ServerUrl {
			conn = c
			break
		}
	}

	if conn == nil {
		return nil, fmt.Errorf("connection not found: %s", req.ServerUrl)
	}

	tools := make([]*pb.Tool, 0, len(conn.Tools))
	for _, tool := range conn.Tools {
		tools = append(tools, tool)
	}

	return &pb.ToolList{Tools: tools}, nil
}

func (m *Manager) CallTool(ctx context.Context, req *pb.CallToolRequest) (*pb.CallToolResponse, error) {
	m.mu.RLock()
	conn, exists := m.connections[req.ConnectionId]
	m.mu.RUnlock()

	if !exists {
		return &pb.CallToolResponse{
			Success: false,
			Error:   fmt.Sprintf("connection not found: %s", req.ConnectionId),
		}, nil
	}

	tool, exists := conn.Tools[req.ToolName]
	if !exists {
		return &pb.CallToolResponse{
			Success: false,
			Error:   fmt.Sprintf("tool not found: %s", req.ToolName),
		}, nil
	}

	// Simulate tool execution
	result := map[string]interface{}{
		"status": "success",
		"tool":   tool.Name,
		"input":  req.Arguments.AsMap(),
	}

	resultStruct, _ := structpb.NewStruct(result)

	return &pb.CallToolResponse{
		Success: true,
		Result:  resultStruct,
	}, nil
}

func (m *Manager) ConnectionCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.connections)
}

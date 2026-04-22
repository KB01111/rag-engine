package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	pb "github.com/ai-engine/proto/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

var (
	grpcAddr = flag.String("addr", "127.0.0.1:50051", "gRPC server address")
	timeout  = flag.Duration("timeout", 20*time.Second, "dial timeout")
)

func main() {
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	conn, err := grpc.NewClient(*grpcAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	runtimeClient := pb.NewRuntimeClient(conn)
	ragClient := pb.NewRagClient(conn)
	trainingClient := pb.NewTrainingClient(conn)
	mcpClient := pb.NewMCPClient(conn)

	fmt.Println("=== AI Engine Client Demo ===")

	// Runtime
	fmt.Println("\n--- Runtime ---")
	status, err := runtimeClient.GetStatus(ctx, &emptypb.Empty{})
	if err != nil {
		log.Printf("GetStatus error: %v", err)
	} else {
		fmt.Printf("Version: %s, Healthy: %v\n", status.Version, status.Healthy)
	}

	models, err := runtimeClient.ListModels(ctx, &emptypb.Empty{})
	if err != nil {
		log.Printf("ListModels error: %v", err)
	} else {
		fmt.Printf("Available models: %d\n", len(models.Models))
	}

	// RAG
	fmt.Println("\n--- RAG ---")
	upsertResp, err := ragClient.UpsertDocument(ctx, &pb.UpsertRequest{
		DocumentId: "doc-001",
		Content:    "This is a sample document about artificial intelligence and machine learning.",
		Metadata:   map[string]string{"title": "AI Introduction", "author": "demo"},
	})
	if err != nil {
		log.Printf("UpsertDocument error: %v", err)
	} else {
		fmt.Printf("Upserted document: %s, chunks: %d\n", upsertResp.DocumentId, upsertResp.ChunksIndexed)
	}

	searchResp, err := ragClient.Search(ctx, &pb.SearchRequest{
		Query:   "artificial intelligence",
		TopK:    5,
		Filters: nil,
	})
	if err != nil {
		log.Printf("Search error: %v", err)
	} else {
		fmt.Printf("Search results: %d, time: %.2fms\n", len(searchResp.Results), searchResp.QueryTimeMs)
		for i, r := range searchResp.Results {
			fmt.Printf("  [%d] Score: %.3f, Text: %s...\n", i+1, r.Score, truncate(r.ChunkText, 50))
		}
	}

	ragStatus, err := ragClient.GetRagStatus(ctx, &emptypb.Empty{})
	if err != nil {
		log.Printf("GetRagStatus error: %v", err)
	} else {
		fmt.Printf("RAG Status - Docs: %d, Chunks: %d\n", ragStatus.DocumentCount, ragStatus.ChunkCount)
	}

	// Training
	fmt.Println("\n--- Training ---")
	run, err := trainingClient.StartRun(ctx, &pb.TrainingRunRequest{
		Name:        "demo-training",
		ModelId:     "llama-7b",
		DatasetPath: "/data/train.jsonl",
		Config:      map[string]string{"epochs": "3"},
	})
	if err != nil {
		log.Printf("StartRun error: %v", err)
	} else {
		fmt.Printf("Started training run: %s (ID: %s)\n", run.Name, run.Id)
	}

	runs, err := trainingClient.ListRuns(ctx, &emptypb.Empty{})
	if err != nil {
		log.Printf("ListRuns error: %v", err)
	} else {
		fmt.Printf("Training runs: %d\n", len(runs.Runs))
	}

	// MCP
	fmt.Println("\n--- MCP ---")
	connResp, err := mcpClient.Connect(ctx, &pb.MCPConnectionRequest{
		ServerUrl: "http://localhost:3000",
		Auth:      map[string]string{},
	})
	if err != nil {
		log.Printf("MCP Connect error: %v", err)
	} else {
		fmt.Printf("MCP connected: %s (ID: %s)\n", connResp.ServerName, connResp.ConnectionId)
	}

	tools, err := mcpClient.ListTools(ctx, &pb.MCPConnectionRequest{
		ServerUrl: "http://localhost:3000",
	})
	if err != nil {
		log.Printf("ListTools error: %v", err)
	} else {
		fmt.Printf("Available tools: %d\n", len(tools.Tools))
		for _, t := range tools.Tools {
			fmt.Printf("  - %s: %s\n", t.Name, t.Description)
		}
	}

	fmt.Println("\n=== Demo Complete ===")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
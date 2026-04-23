package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	pb "github.com/ai-engine/proto/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

var (
	grpcAddr   = flag.String("addr", "127.0.0.1:50051", "gRPC server address")
	timeout    = flag.Duration("timeout", 20*time.Second, "dial timeout")
	modelID    = flag.String("model-id", "", "model to load before inference; defaults to the first discovered model")
	prompt     = flag.String("prompt", "Summarize the local AI engine status in one sentence.", "prompt to send to runtime inference")
	docID      = flag.String("doc-id", "doc-winui-smoke", "document id to upsert or verify")
	docContent = flag.String("doc-content", "Artificial intelligence systems can combine retrieval, local models, and persistent storage.", "document content to upsert")
	query      = flag.String("query", "retrieval local models", "search query to execute against RAG")
	skipUpsert = flag.Bool("skip-upsert", false, "skip document upsert and only verify/search existing data")
)

func main() {
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	conn, err := grpc.NewClient(*grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	runtimeClient := pb.NewRuntimeClient(conn)
	ragClient := pb.NewRagClient(conn)

	fmt.Println("=== AI Engine Runtime + RAG Demo ===")

	status, err := runtimeClient.GetStatus(ctx, &emptypb.Empty{})
	if err != nil {
		log.Fatalf("GetStatus error: %v", err)
	}
	fmt.Printf("Version: %s, Healthy: %v\n", status.Version, status.Healthy)
	fmt.Printf(
		"Resources: cpu=%.2f%% memory=%d/%d\n",
		status.Resources.CpuPercent,
		status.Resources.MemoryUsedBytes,
		status.Resources.MemoryTotalBytes,
	)

	models, err := runtimeClient.ListModels(ctx, &emptypb.Empty{})
	if err != nil {
		log.Fatalf("ListModels error: %v", err)
	}
	fmt.Printf("Available models: %d\n", len(models.Models))
	selectedModel := chooseModel(models.Models, *modelID)
	if selectedModel == "" {
		log.Fatal("No model available to load")
	}

	loaded, err := runtimeClient.LoadModel(ctx, &pb.LoadModelRequest{ModelId: selectedModel})
	if err != nil {
		log.Fatalf("LoadModel error: %v", err)
	}
	fmt.Printf("Loaded model: %s\n", loaded.Id)

	inference, err := runtimeClient.StreamInference(ctx)
	if err != nil {
		log.Fatalf("StreamInference open error: %v", err)
	}
	if err := inference.Send(&pb.InferenceRequest{
		ModelId:    selectedModel,
		Prompt:     *prompt,
		Parameters: map[string]string{"n_predict": "64"},
	}); err != nil {
		log.Fatalf("StreamInference send error: %v", err)
	}
	if err := inference.CloseSend(); err != nil {
		log.Fatalf("StreamInference close error: %v", err)
	}

	var inferenceOutput strings.Builder
	for {
		resp, err := inference.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("StreamInference recv error: %v", err)
		}
		if resp.Token != "" {
			inferenceOutput.WriteString(resp.Token)
		}
		if resp.Complete {
			break
		}
	}
	fmt.Printf("Inference output: %s\n", truncate(strings.TrimSpace(inferenceOutput.String()), 160))

	if !*skipUpsert {
		upsertResp, err := ragClient.UpsertDocument(ctx, &pb.UpsertRequest{
			DocumentId: *docID,
			Content:    *docContent,
			Metadata: map[string]string{
				"title":       "WinUI Runtime Smoke Doc",
				"source_db":   "local",
				"external_id": *docID,
			},
		})
		if err != nil {
			log.Fatalf("UpsertDocument error: %v", err)
		}
		fmt.Printf("Upserted document: %s, chunks: %d\n", upsertResp.DocumentId, upsertResp.ChunksIndexed)
	}

	documents, err := ragClient.ListDocuments(ctx, &emptypb.Empty{})
	if err != nil {
		log.Fatalf("ListDocuments error: %v", err)
	}
	fmt.Printf("Documents: %d\n", len(documents.Documents))
	present := false
	for _, doc := range documents.Documents {
		if doc.Id == *docID {
			present = true
			break
		}
	}
	fmt.Printf("Document present: %v\n", present)

	searchResp, err := ragClient.Search(ctx, &pb.SearchRequest{
		Query:   *query,
		TopK:    5,
		Filters: map[string]string{"external_id": *docID},
	})
	if err != nil {
		log.Fatalf("Search error: %v", err)
	}
	fmt.Printf("Search results: %d, time: %.2fms\n", len(searchResp.Results), searchResp.QueryTimeMs)

	fmt.Println("=== Demo Complete ===")
}

func chooseModel(models []*pb.ModelInfo, preferred string) string {
	if preferred != "" {
		return preferred
	}
	if len(models) == 0 {
		return ""
	}
	return models[0].Id
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

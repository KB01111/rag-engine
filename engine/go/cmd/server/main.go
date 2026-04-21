package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ai-engine/go/internal/api"
	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/supervisor"
	"log"
	"net/http"
)

var (
	version     = "1.0.0"
	configPath  = flag.String("config", "", "path to config file")
	showVersion = flag.Bool("version", false, "show version")
)

func main() {
	flag.Parse()

	if *showVersion {
		fmt.Printf("AI Engine v%s\n", version)
		os.Exit(0)
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	sup := supervisor.NewSupervisor(cfg)
	if err := sup.Start(); err != nil {
		log.Fatalf("Failed to start supervisor: %v", err)
	}

	server := api.NewServer(cfg, sup)

	errCh := make(chan error, 2)

	// Start HTTP server
	go func() {
		addr := cfg.Addr()
		log.Printf("Starting HTTP server on %s", addr)
		if err := server.StartHTTP(addr); err != nil && err != http.ErrServerClosed {
			errCh <- fmt.Errorf("HTTP server error: %w", err)
		}
	}()

	// Start gRPC server
	go func() {
		addr := cfg.GRPCAddr()
		log.Printf("Starting gRPC server on %s", addr)
		if err := server.StartGRPC(addr); err != nil {
			errCh <- fmt.Errorf("gRPC server error: %w", err)
		}
	}()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		log.Printf("Server error: %v", err)
	case sig := <-sigCh:
		log.Printf("Received signal: %v", sig)
	}

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	log.Println("Shutting down...")
	server.Stop()
	sup.Stop()

	_ = ctx // suppress unused warning
	log.Println("Server stopped")
}

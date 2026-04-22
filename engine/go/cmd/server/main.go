package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ai-engine/go/internal/api"
	"github.com/ai-engine/go/internal/config"
	"github.com/ai-engine/go/internal/supervisor"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"net/http"
)

var (
	version     = "1.0.0"
	configPath  = flag.String("config", "", "path to config file")
	showVersion = flag.Bool("version", false, "show version")
)

func main() {
	flag.Parse()

	logger := zerolog.New(os.Stdout).With().Timestamp().Caller().Logger()
	zerolog.SetGlobalLevel(zerolog.InfoLevel)

	if *showVersion {
		fmt.Printf("AI Engine v%s\n", version)
		os.Exit(0)
	}

	logger.Info().Msg("Starting AI Engine")

	cfg, err := config.Load(*configPath)
	if err != nil {
		logger.Fatal().Err(err).Msg("Failed to load config")
	}

	sup := supervisor.NewSupervisor(cfg)
	if err := sup.Start(); err != nil {
		logger.Fatal().Err(err).Msg("Failed to start supervisor")
	}

	server := api.NewServer(cfg, sup, logger)

	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())
	server.RegisterHTTP(router)

	// Buffered with capacity 2 so both HTTP and gRPC server goroutines can report errors
	// without blocking, even though main only receives once before shutting down.
	errCh := make(chan error, 2)

	go func() {
		addr := cfg.Addr()
		logger.Info().Str("addr", addr).Msg("Starting HTTP server")
		if err := server.StartHTTP(addr, router); err != nil && err != http.ErrServerClosed {
			errCh <- fmt.Errorf("HTTP server error: %w", err)
		}
	}()

	go func() {
		addr := cfg.GRPCAddr()
		logger.Info().Str("addr", addr).Msg("Starting gRPC server")
		if err := server.StartGRPC(addr); err != nil {
			errCh <- fmt.Errorf("gRPC server error: %w", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		logger.Error().Err(err).Msg("Server error")
	case sig := <-sigCh:
		logger.Info().Str("signal", sig.String()).Msg("Received signal")
	}

	logger.Info().Msg("Shutting down...")
	server.Stop()
	sup.Stop()

	logger.Info().Msg("Server stopped")
}

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	pb "github.com/ai-engine/proto/go"
	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"
)

type loadModelHTTPBody struct {
	ModelID string            `json:"model_id"`
	Options map[string]string `json:"options,omitempty"`
}

type unloadModelHTTPBody struct {
	ModelID string `json:"model_id"`
}

type inferenceHTTPBody struct {
	ModelID     string            `json:"model_id"`
	Provider    string            `json:"provider,omitempty"`
	Prompt      string            `json:"prompt"`
	Parameters  map[string]string `json:"parameters,omitempty"`
	ContextRefs []string          `json:"context_refs,omitempty"`
}

func (s *Server) handleRuntimeStatus(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	status, err := s.supervisor.Runtime.GetStatus(ctx, &emptypb.Empty{})
	if err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Msg("runtime status failed")
		backendError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":         status,
		"execution_mode": s.supervisor.ExecutionMode(),
	})
}

func (s *Server) handleLoadRuntimeModel(c *gin.Context) {
	var req loadModelHTTPBody
	if err := c.ShouldBindJSON(&req); err != nil {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, err.Error())
		return
	}
	req.ModelID = strings.TrimSpace(req.ModelID)
	if req.ModelID == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "model_id is required")
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	model, err := s.supervisor.Runtime.LoadModel(ctx, &pb.LoadModelRequest{
		ModelId: req.ModelID,
		Options: req.Options,
	})
	if err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Str("model_id", req.ModelID).Msg("load model failed")
		backendError(c, err)
		return
	}

	c.JSON(http.StatusOK, model)
}

func (s *Server) handleUnloadRuntimeModel(c *gin.Context) {
	var req unloadModelHTTPBody
	if err := c.ShouldBindJSON(&req); err != nil {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, err.Error())
		return
	}
	req.ModelID = strings.TrimSpace(req.ModelID)
	if req.ModelID == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "model_id is required")
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	if _, err := s.supervisor.Runtime.UnloadModel(ctx, &pb.UnloadModelRequest{ModelId: req.ModelID}); err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Str("model_id", req.ModelID).Msg("unload model failed")
		backendError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"model_id": req.ModelID,
		"unloaded": true,
	})
}

func (s *Server) handleStreamRuntimeInference(c *gin.Context) {
	var req inferenceHTTPBody
	if err := c.ShouldBindJSON(&req); err != nil {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, err.Error())
		return
	}
	req.ModelID = strings.TrimSpace(req.ModelID)
	req.Prompt = strings.TrimSpace(req.Prompt)
	if req.ModelID == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "model_id is required")
		return
	}
	if req.Prompt == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "prompt is required")
		return
	}

	stream := &httpInferenceStream{
		ctx:    c.Request.Context(),
		writer: c.Writer,
		req: &pb.InferenceRequest{
			ModelId:     req.ModelID,
			Provider:    req.Provider,
			Prompt:      req.Prompt,
			Parameters:  req.Parameters,
			ContextRefs: req.ContextRefs,
		},
	}

	err := s.supervisor.Runtime.StreamInference(c.Request.Context(), stream)
	if err == nil {
		return
	}

	s.log.Warn().Err(err).Str("request_id", requestID(c)).Str("model_id", req.ModelID).Bool("stream_started", stream.started).Msg("stream inference failed")
	if stream.started {
		_ = stream.sendError(apiErrorBackendUnavailable, err.Error())
		return
	}
	backendError(c, err)
}

type httpInferenceStream struct {
	ctx    context.Context
	writer gin.ResponseWriter
	req    *pb.InferenceRequest

	received bool
	started  bool
}

func (s *httpInferenceStream) Context() context.Context {
	return s.ctx
}

func (s *httpInferenceStream) Recv() (*pb.InferenceRequest, error) {
	if s.received {
		return nil, io.EOF
	}
	s.received = true
	return s.req, nil
}

func (s *httpInferenceStream) Send(resp *pb.InferenceResponse) error {
	event := "token"
	if resp.GetComplete() {
		event = "complete"
	}
	return s.writeEvent(event, resp)
}

func (s *httpInferenceStream) SetHeader(metadata.MD) error {
	return nil
}

func (s *httpInferenceStream) SendHeader(metadata.MD) error {
	return nil
}

func (s *httpInferenceStream) SetTrailer(metadata.MD) {}

func (s *httpInferenceStream) SendMsg(any) error {
	return nil
}

func (s *httpInferenceStream) RecvMsg(any) error {
	return nil
}

func (s *httpInferenceStream) sendError(code, message string) error {
	return s.writeEvent("error", apiErrorBody{
		Error: apiError{
			Code:    code,
			Message: message,
		},
	})
}

func (s *httpInferenceStream) writeEvent(event string, payload any) error {
	if !s.started {
		header := s.writer.Header()
		header.Set("Content-Type", "text/event-stream")
		header.Set("Cache-Control", "no-cache")
		header.Set("Connection", "keep-alive")
		s.writer.WriteHeader(http.StatusOK)
		s.started = true
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.writer, "event: %s\ndata: %s\n\n", event, raw); err != nil {
		return err
	}
	s.writer.Flush()
	return nil
}

func requestID(c *gin.Context) string {
	if value, ok := c.Get("request_id"); ok {
		if requestID, ok := value.(string); ok {
			return requestID
		}
	}
	return c.GetHeader(requestIDHeader)
}

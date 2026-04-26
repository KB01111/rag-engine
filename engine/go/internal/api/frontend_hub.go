package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/ai-engine/go/internal/config"
	hubsvc "github.com/ai-engine/go/internal/hub"
	"github.com/gin-gonic/gin"
)

type RuntimeHubService interface {
	Search(context.Context, hubsvc.SearchRequest) (hubsvc.SearchResponse, error)
	Model(context.Context, string) (hubsvc.Model, error)
	StartDownload(context.Context, hubsvc.DownloadRequest) (hubsvc.Download, error)
	ListDownloads() []hubsvc.Download
	GetDownload(string) (hubsvc.Download, bool)
	CancelDownload(string) (hubsvc.Download, error)
}

func newRuntimeHubService(cfg *config.Config) RuntimeHubService {
	if cfg == nil || !cfg.HuggingFace.Enabled {
		return nil
	}
	return hubsvc.NewService(hubsvc.Config{
		Endpoint:             cfg.HuggingFace.Endpoint,
		ModelsPath:           cfg.Runtime.ModelsPath,
		MaxDownloadBytes:     cfg.HuggingFace.MaxDownloadBytes,
		CompatibleExtensions: cfg.HuggingFace.CompatibleExtensions,
	})
}

// SetRuntimeHubService is test-only; called before handlers access s.hub.
func (s *Server) SetRuntimeHubService(service RuntimeHubService) {
	s.hub = service
}

func (s *Server) handleRuntimeHubSearch(c *gin.Context) {
	if s.hub == nil {
		writeAPIError(c, http.StatusServiceUnavailable, apiErrorBackendUnavailable, "Hugging Face hub integration is disabled")
		return
	}

	limit := 20
	if rawLimit := strings.TrimSpace(c.Query("limit")); rawLimit != "" {
		parsed, err := strconv.Atoi(rawLimit)
		if err != nil || parsed <= 0 {
			writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "limit must be a positive integer")
			return
		}
		if parsed > 100 {
			writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "limit must be between 1 and 100")
			return
		}
		limit = parsed
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 15*time.Second)
	defer cancel()

	searchReq := hubsvc.SearchRequest{
		Query:               c.Query("query"),
		Author:              c.Query("author"),
		Sort:                c.Query("sort"),
		Limit:               limit,
		Cursor:              c.Query("cursor"),
		IncludeIncompatible: queryBool(c, "include_incompatible"),
	}
	result, err := s.hub.Search(ctx, searchReq)
	if err != nil {
		s.writeHubError(c, err)
		return
	}
	result.Query = strings.TrimSpace(searchReq.Query)
	result.Author = strings.TrimSpace(searchReq.Author)
	result.Sort = strings.TrimSpace(searchReq.Sort)
	result.Limit = searchReq.Limit
	result.CompatibleOnly = !searchReq.IncludeIncompatible
	c.JSON(http.StatusOK, result)
}

func (s *Server) handleRuntimeHubModel(c *gin.Context) {
	if s.hub == nil {
		writeAPIError(c, http.StatusServiceUnavailable, apiErrorBackendUnavailable, "Hugging Face hub integration is disabled")
		return
	}
	repoID := strings.TrimSpace(c.Query("repo_id"))
	if repoID == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "repo_id is required")
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 15*time.Second)
	defer cancel()

	model, err := s.hub.Model(ctx, repoID)
	if err != nil {
		s.writeHubError(c, err)
		return
	}
	c.JSON(http.StatusOK, model)
}

func (s *Server) handleRuntimeHubStartDownload(c *gin.Context) {
	if s.hub == nil {
		writeAPIError(c, http.StatusServiceUnavailable, apiErrorBackendUnavailable, "Hugging Face hub integration is disabled")
		return
	}

	var req hubsvc.DownloadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, err.Error())
		return
	}

	download, err := s.hub.StartDownload(c.Request.Context(), req)
	if err != nil {
		s.writeHubError(c, err)
		return
	}
	c.JSON(http.StatusAccepted, download)
}

func (s *Server) handleRuntimeHubListDownloads(c *gin.Context) {
	if s.hub == nil {
		writeAPIError(c, http.StatusServiceUnavailable, apiErrorBackendUnavailable, "Hugging Face hub integration is disabled")
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"downloads": s.hub.ListDownloads(),
	})
}

func (s *Server) handleRuntimeHubDownloadEvents(c *gin.Context) {
	if s.hub == nil {
		writeAPIError(c, http.StatusServiceUnavailable, apiErrorBackendUnavailable, "Hugging Face hub integration is disabled")
		return
	}
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "download id is required")
		return
	}
	if _, ok := s.hub.GetDownload(id); !ok {
		writeAPIError(c, http.StatusNotFound, apiErrorNotFound, "download not found")
		return
	}

	header := c.Writer.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	c.Writer.WriteHeader(http.StatusOK)

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	var lastStatus string
	var lastBytes int64 = -1
	for {
		download, ok := s.hub.GetDownload(id)
		if !ok {
			_ = writeHubDownloadEvent(c.Writer, "error", apiErrorBody{Error: apiError{Code: apiErrorNotFound, Message: "download not found"}})
			return
		}
		if download.Status != lastStatus || download.DownloadedBytes != lastBytes || download.Terminal() {
			event := downloadEventName(download.Status)
			_ = writeHubDownloadEvent(c.Writer, event, download)
			lastStatus = download.Status
			lastBytes = download.DownloadedBytes
		}
		if download.Terminal() {
			return
		}

		select {
		case <-c.Request.Context().Done():
			return
		case <-ticker.C:
		}
	}
}

func downloadEventName(status string) string {
	switch status {
	case hubsvc.DownloadStatusCompleted:
		return "complete"
	case hubsvc.DownloadStatusFailed:
		return "error"
	case hubsvc.DownloadStatusCanceled:
		return "canceled"
	default:
		return "progress"
	}
}

func (s *Server) handleRuntimeHubCancelDownload(c *gin.Context) {
	if s.hub == nil {
		writeAPIError(c, http.StatusServiceUnavailable, apiErrorBackendUnavailable, "Hugging Face hub integration is disabled")
		return
	}
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "download id is required")
		return
	}

	download, err := s.hub.CancelDownload(id)
	if err != nil {
		s.writeHubError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"canceled": true,
		"download": download,
	})
}

func (s *Server) writeHubError(c *gin.Context, err error) {
	code := hubsvc.Code(err)
	message := hubsvc.Message(err)
	switch code {
	case hubsvc.ErrorCodeInvalidRequest:
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, message)
	case hubsvc.ErrorCodeUnsupportedModelFormat:
		writeAPIError(c, http.StatusBadRequest, hubsvc.ErrorCodeUnsupportedModelFormat, message)
	case hubsvc.ErrorCodeNotFound:
		writeAPIError(c, http.StatusNotFound, apiErrorNotFound, message)
	case hubsvc.ErrorCodeDownloadTooLarge:
		writeAPIError(c, http.StatusRequestEntityTooLarge, hubsvc.ErrorCodeDownloadTooLarge, message)
	case hubsvc.ErrorCodeHuggingFaceTokenMissing:
		writeAPIError(c, http.StatusForbidden, hubsvc.ErrorCodeHuggingFaceTokenMissing, message)
	default:
		writeAPIError(c, http.StatusBadGateway, apiErrorBackendUnavailable, message)
	}
}

func writeHubDownloadEvent(writer gin.ResponseWriter, event string, payload any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := writer.Write([]byte("event: " + event + "\n")); err != nil {
		return err
	}
	if _, err := writer.Write([]byte("data: " + string(raw) + "\n\n")); err != nil {
		return err
	}
	writer.Flush()
	return nil
}

func queryBool(c *gin.Context, key string) bool {
	switch strings.ToLower(strings.TrimSpace(c.Query(key))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

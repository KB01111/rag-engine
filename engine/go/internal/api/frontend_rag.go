package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	pb "github.com/ai-engine/proto/go"
	"github.com/gin-gonic/gin"
	"google.golang.org/protobuf/types/known/emptypb"
)

func (s *Server) handleRAGStatus(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	status, err := s.supervisor.RAG.GetRagStatus(ctx, &emptypb.Empty{})
	if err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Msg("rag status failed")
		backendError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"status":         status,
		"execution_mode": s.supervisor.ExecutionMode(),
	})
}

func (s *Server) handleListRAGDocuments(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	documents, err := s.supervisor.RAG.ListDocuments(ctx, &emptypb.Empty{})
	if err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Msg("list rag documents failed")
		backendError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"documents": documents.Documents})
}

func (s *Server) handleUpsertRAGDocument(c *gin.Context) {
	var req pb.UpsertRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.GetContent()) == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "content is required")
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	resp, err := s.supervisor.RAG.UpsertDocument(ctx, &req)
	if err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Str("document_id", req.GetDocumentId()).Msg("upsert rag document failed")
		backendError(c, err)
		return
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleDeleteRAGDocument(c *gin.Context) {
	documentID := strings.TrimSpace(c.Param("id"))
	if documentID == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "document id is required")
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	if _, err := s.supervisor.RAG.DeleteDocument(ctx, &pb.DeleteRequest{DocumentId: documentID}); err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Str("document_id", documentID).Msg("delete rag document failed")
		backendError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"document_id": documentID,
		"deleted":     true,
	})
}

func (s *Server) handleSearchRAG(c *gin.Context) {
	var req pb.SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.GetQuery()) == "" {
		writeAPIError(c, http.StatusBadRequest, apiErrorInvalidRequest, "query is required")
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 20*time.Second)
	defer cancel()

	resp, err := s.supervisor.RAG.Search(ctx, &req)
	if err != nil {
		s.log.Warn().Err(err).Str("request_id", requestID(c)).Msg("rag search failed")
		backendError(c, err)
		return
	}
	c.JSON(http.StatusOK, resp)
}

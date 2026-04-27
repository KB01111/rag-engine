package api

import (
	"context"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/metadata"
)

func withRequestID(ctx context.Context, c *gin.Context) context.Context {
	id := requestID(c)
	if id == "" {
		return ctx
	}
	return metadata.AppendToOutgoingContext(ctx, "x-request-id", id)
}

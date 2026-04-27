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
	ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", id)
	return metadata.NewIncomingContext(ctx, metadata.Pairs("x-request-id", id))
}

package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/metadata"
)

func TestHTTPInferenceStreamSerializesEventsAndHonorsCancel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ctx, cancel := context.WithCancel(context.Background())
	stream := &httpInferenceStream{
		ctx:    ctx,
		writer: ginCtx.Writer,
	}

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		errs <- stream.writeEvent("token", map[string]string{"token": "hello"})
	}()
	go func() {
		defer wg.Done()
		errs <- stream.writeComment("keepalive")
	}()
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("write failed: %v", err)
		}
	}

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected stream to start with 200, got %d", recorder.Code)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, "event: token") {
		t.Fatalf("missing token event: %q", body)
	}
	if !strings.Contains(body, ":keepalive\n\n") {
		t.Fatalf("missing keepalive comment: %q", body)
	}

	cancel()
	if err := stream.writeEvent("token", map[string]string{"token": "late"}); err == nil {
		t.Fatal("expected canceled context error")
	}
}

func TestWithRequestIDAttachesIncomingAndOutgoingMetadata(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ginCtx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ginCtx.Set("request_id", "rid-42")

	ctx := withRequestID(context.Background(), ginCtx)

	incoming, ok := metadata.FromIncomingContext(ctx)
	if !ok || len(incoming.Get("x-request-id")) != 1 || incoming.Get("x-request-id")[0] != "rid-42" {
		t.Fatalf("incoming request id not attached: %v", incoming)
	}
	outgoing, ok := metadata.FromOutgoingContext(ctx)
	if !ok || len(outgoing.Get("x-request-id")) != 1 || outgoing.Get("x-request-id")[0] != "rid-42" {
		t.Fatalf("outgoing request id not attached: %v", outgoing)
	}
}

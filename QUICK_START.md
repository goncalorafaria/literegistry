# Quick Start: New LiteLLM-Style Gateway

## 🚨 What Changed?

**OLD ARCHITECTURE** (File descriptor leak):
- 128 pooled clients per model
- Each client had 100-connection pool
- Total: 12,800 potential file descriptors ❌

**NEW ARCHITECTURE** (LiteLLM pattern):
- 1 shared aiohttp session for entire app
- Unlimited connections with 2-min keep-alive
- File descriptors: Only what's actively needed ✅

## 🚀 How to Start the New Gateway

###Step 1: Increase File Descriptor Limit
```bash
# Temporary (current session)
ulimit -n 65536

# Permanent (add to ~/.bashrc or /etc/security/limits.conf)
echo "ulimit -n 65536" >> ~/.bashrc
```

### Step 2: Stop Old Gateway
```bash
# Find and kill old gateway processes
pkill -f "literegistry.gateway"

# Verify stopped
pgrep -f "literegistry.gateway" || echo "All stopped ✅"
```

### Step 3: Start New Gateway

**Production Mode** (recommended):
```bash
ulimit -n 65536 && \
uvicorn literegistry.gateway:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4
```

**Development Mode**:
```bash
ulimit -n 65536 && \
python -m literegistry.gateway \
  --registry='redis://klone-login03.hyak.local:6379' \
  --port=8080
```

### Step 4: Verify It's Working

**Check health**:
```bash
curl http://localhost:8080/health | jq
```

**Check shared session** (NEW endpoint):
```bash
curl http://localhost:8080/session-stats | jq
```

Expected output:
```json
{
  "status": "success",
  "session_info": {
    "shared_session_initialized": true,
    "architecture": "single_shared_session",
    "pattern": "litellm_style",
    "connector_limit": 0,
    "keepalive_timeout": 120,
    "dns_cache_ttl": 300
  }
}
```

## 📊 Monitoring

### Watch File Descriptors
```bash
# Get gateway PID
GATEWAY_PID=$(pgrep -f "uvicorn.*literegistry.gateway" | head -1)
echo "Gateway PID: $GATEWAY_PID"

# Watch file descriptor count
watch -n 1 "ls /proc/$GATEWAY_PID/fd 2>/dev/null | wc -l"
```

**Expected**: Should stay low (< 500) even under load

### Watch Connections
```bash
# Show active connections by state
watch -n 1 "ss -tnp 2>/dev/null | grep $GATEWAY_PID | awk '{print \$1}' | sort | uniq -c"
```

**Expected**: Mostly ESTABLISHED, some TIME-WAIT

## 🎯 What to Expect

### ✅ Good Signs
- File descriptors stay stable (< 500)
- No "too many open files" errors
- Logs show "✅ Gateway started with shared aiohttp session"
- `/session-stats` shows `shared_session_initialized: true`
- Requests complete successfully

### ⚠️ Warning Signs  
- File descriptors growing rapidly
- "too many open files" errors
- Session stats show `shared_session_initialized: false`
- Warnings about "temporary session" in logs

If you see warnings, check that:
1. ulimit is set correctly (`ulimit -n` should show 65536)
2. Gateway restarted (old processes killed)
3. No errors in startup logs

## 📝 API Changes

### Endpoints (mostly unchanged)
- ✅ `GET /health` - Health check (same)
- ✅ `GET /v1/models` - List models (same)
- ✅ `POST /v1/completions` - Completions (same)
- ✅ `POST /classify` - Classification (same)
- 🆕 `GET /session-stats` - Session info (NEW, replaces `/pool-stats`)

### Request/Response Format
**No changes!** Same JSON format for all endpoints.

## 🐛 Troubleshooting

### Problem: "too many open files"
**Solution**:
```bash
# Check current limit
ulimit -n

# Increase if needed
ulimit -n 65536

# Restart gateway
pkill -f "literegistry.gateway"
# ... start command ...
```

### Problem: "Shared session not available" warnings
**Cause**: Gateway not restarted with new code

**Solution**:
```bash
# Kill all old processes
pkill -9 -f "literegistry.gateway"

# Restart with new code
ulimit -n 65536 && uvicorn literegistry.gateway:app --host 0.0.0.0 --port 8080
```

### Problem: High file descriptor count
**Check**: Are you running multiple gateway instances?

```bash
# List all gateway processes
pgrep -af "literegistry.gateway"

# Kill duplicates
pkill -f "literegistry.gateway"

# Start fresh
ulimit -n 65536 && uvicorn literegistry.gateway:app --host 0.0.0.0 --port 8080 --workers 4
```

## 🧪 Test Under Load

```bash
# Simple load test (requires apache-bench)
ab -n 1000 -c 50 -T 'application/json' -p request.json http://localhost:8080/v1/completions

# Watch file descriptors during test
watch -n 0.5 'ls /proc/'$(pgrep -f "uvicorn.*literegistry" | head -1)'/fd | wc -l'
```

**Expected**: FDs stay low even during high load

## 📚 More Info

- **Full Implementation Plan**: See `LITELLM_IMPLEMENTATION_PLAN.md`
- **LiteLLM Comparison**: See `LITELLM_COMPARISON.md`  
- **Summary**: See `GATEWAY_IMPROVEMENTS_SUMMARY.md`

## ✅ Success Checklist

Before considering deployment complete:

- [ ] Old gateway processes stopped
- [ ] ulimit set to 65536
- [ ] New gateway started successfully
- [ ] `/health` endpoint responds
- [ ] `/session-stats` shows `shared_session_initialized: true`
- [ ] File descriptors stay low under test load
- [ ] No "too many open files" errors
- [ ] Requests completing successfully

---

**Architecture**: LiteLLM-style shared session ✅  
**File Descriptor Issue**: RESOLVED ✅  
**Status**: Ready for production 🚀


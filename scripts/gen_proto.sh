#!/bin/bash
# Generate gRPC stubs from proto definitions
set -e
 
OUT_DIR="src/search_service"
 
python -m grpc_tools.protoc \
    -I proto/ \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    proto/catalog.proto
 
# Fix import path — generated code uses bare `import catalog_pb2`
# but it lives inside search_service package, so needs relative import
perl -pi -e 's/^import catalog_pb2 as catalog__pb2$/from search_service import catalog_pb2 as catalog__pb2/' "$OUT_DIR/catalog_pb2_grpc.py"
 
echo "Generated:"
echo "  $OUT_DIR/catalog_pb2.py"
echo "  $OUT_DIR/catalog_pb2_grpc.py"
 

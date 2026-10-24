import os
import tempfile

# Isolate every test run from the real data/ directory (must happen before threeddd is imported).
os.environ["THREEDDD_DATA"] = tempfile.mkdtemp(prefix="threeddd-test-")

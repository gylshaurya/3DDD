import os
import tempfile

# Isolate every test run from the real data/ directory (must happen before splatcast is imported).
os.environ["SPLATCAST_DATA"] = tempfile.mkdtemp(prefix="splatcast-test-")

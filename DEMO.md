# 3DDD demo playbook

## Before you go on stage (10 min)

1. Plug in the laptop and close other apps. The 16 GB of RAM is shared with the GPU.
2. `./run.sh`, then open **http://localhost:8000** in Chrome. Check that the hero scene is orbiting smoothly.
3. Check the **GPU budget** badge (top right). It should be well under the cap.
4. **Warm up Modal**: run the sample once with *Preview · L4* (about $0.03). This makes sure the token works
   and the GPU image is cached. Delete the result from its scene page afterwards if you like.
5. Open a replay in a second tab: **http://localhost:8000/#/replay/1025-014849-4bfe** (the Bonsai run,
   30.7 dB). Every finished job page also has a "Replay training" button. That's your fallback if the venue
   Wi-Fi fails.
6. Phone upload only works if the phone and laptop are on the same network and the venue Wi-Fi doesn't
   isolate clients. Test it once. If it fails, AirDrop the video to the laptop and drag it onto the page.

## The 3-minute pitch

| Time | Say | Do |
|---|---|---|
| 0:00 | "Photos are flat. What if a 30-second phone video gave you a place you can walk around inside?" | Home page, hero scene orbiting behind you |
| 0:15 | "Let me show you, live." | Drop the video (or scan the QR and film the judges' table) · *Fast · L40S* (no video? click a sample: counter is the fastest) |
| 0:25 | "First we pick the sharpest frames. Motion blur is the main thing that breaks 3D reconstruction." | Contact sheet appears |
| 0:35 | "Now we solve where the phone was for every single frame: SIFT features, matching, then a global structure-from-motion solver." | Point at the timeline and log |
| 1:30 | "Each gold pyramid is a phone position we recovered. Sub-pixel accuracy." | The SfM view with gold frustums |
| 1:45 | "Now a cloud GPU starts optimising hundreds of thousands of 3D Gaussians, and it streams the half-trained scene back to the browser while it learns." | The blurry blobs sharpen; PSNR curve climbs |
| 2:30 | "That's rendered by a WebGPU renderer we wrote ourselves, including a GPU radix sort of every Gaussian, every frame, at 60 fps." | Point at the fps / sort-ms HUD |
| 2:45 | "And to prove it isn't just memorising the video: these are camera angles it never saw. Real photo on the left, our render on the right." | Held-out comparisons panel |
| 2:55 | Open the scene, turn on fly mode, fly into it. | |

**If something breaks:** say "here's one I captured earlier", open the replay tab, and keep talking.
The replay looks exactly like a live run.

## Questions judges will ask

**What is Gaussian Splatting?**
The scene is represented as hundreds of thousands of small, semi-transparent 3D ellipsoids. Each has a
position, a shape (3D covariance from scale and rotation), an opacity and a view-dependent colour (spherical
harmonics). To render, you project each one to a 2D ellipse on screen, sort them by depth, and alpha-blend
front to back. Training is gradient descent: render a training view, compare it with the real photo, and
backpropagate into every Gaussian's parameters.

**Why not NeRF?**
NeRF stores the scene inside a neural network and needs hundreds of network queries per pixel, so it's slow
to render. Splats are explicit geometry that rasterises in real time, even in a browser.

**What did you actually build vs. use?**
- *Built:* the whole WebGPU renderer (projection maths, SH shading, GPU radix sort, culling, streaming loader,
  camera). The training loop (initialisation, MCMC densification schedule, loss, the live snapshot
  streaming, the compact transfer format). Frame selection. Gravity alignment and framing. The job system with
  live SSE streaming. The spending guard. The whole UI.
- *Used:* gsplat's CUDA rasterisation kernels for the backward pass, COLMAP/GLOMAP (via pycolmap) for
  structure-from-motion, Brush for the offline Mac trainer, Modal for GPUs.

**How do you get camera positions from a video?**
Structure-from-motion. Detect SIFT keypoints in each frame, match them between nearby frames, keep geometrically
consistent matches, then solve for all camera poses and 3D points at once. We use a *global* solver (GLOMAP),
which first estimates all rotations and then all positions together, and falls back to incremental SfM if needed.

**Why sort on the GPU?**
Alpha blending needs back-to-front order, and the order changes every time the camera moves. Sorting 600k keys
on the CPU each frame would cap the frame rate. Our WGSL radix sort does it in about 1-2 ms.

**What's MCMC densification?**
3DGS has to add Gaussians where detail is missing. The original heuristic (clone/split on gradient size) can
blow up memory. The MCMC variant treats Gaussians as samples. It "teleports" dead, low-opacity ones to where
they're needed and grows up to a hard cap, so memory and runtime stay predictable. That matters for a live
demo.

**How do you handle drift when walking around an object?**
Matching only neighbouring frames lets small errors accumulate around the loop, so the end of the walk
doesn't line up with the start and you get a doubled "ghost" object. We match every frame against its
neighbours *and* against 15 keyframes spread across the whole capture, which closes the loop. On the kitchen
scene that took held-out PSNR from 25.5 to 29.9 dB.

**What does 26.7 dB PSNR mean?**
PSNR measures how closely our render matches the real photo, pixel by pixel. We hold out every 10th frame, never
train on it, and score renders from exactly those viewpoints. Around 27 dB after 90 seconds of training is a
strong result for this scene; the original paper reports about 29 dB on the same scene after 30k steps.

**Where does it fail?**
Mirrors, glass and shiny surfaces (they look different from every angle). Moving objects. Plain textureless
walls (nothing for SIFT to match). Fast motion blur. Regions the camera never saw.

**Cost?**
About $0.15 (fast) to $0.35 (quality) of GPU time per scene on an L40S. A hard budget guard checks real Modal billing before any GPU starts.

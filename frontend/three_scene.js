/**
 * three_scene.js — 3D Neural Network Visualization using Three.js
 * Renders glowing neuron layers with animated data-flow connections,
 * inspired by the reference images: cyan dots on dark background,
 * with colored (green/red/purple) animated lines flowing between layers.
 */

(function () {
  'use strict';

  // ─── Scene State ──────────────────────────────────────────────
  let scene, camera, renderer, controls;
  let animFrameId = null;
  let autoRotate  = true;
  let isAnimating = false;  // true while inference animation plays

  // Layer groups
  const layerGroups = [];
  const connectionGroups = [];

  // Neuron meshes per layer (for highlight)
  const layerNeurons = [];

  // Inference animation state
  let inferenceWave   = 0;
  let inferenceActive = false;
  let inferenceScores = [];

  // Color palette
  const COL_CYAN   = new THREE.Color(0x00e5ff);
  const COL_PURPLE = new THREE.Color(0x9933ff);
  const COL_GREEN  = new THREE.Color(0x00ff9d);
  const COL_RED    = new THREE.Color(0xff4d6d);
  const COL_DIM    = new THREE.Color(0x0a1a2a);

  // ─── Layer definitions ────────────────────────────────────────
  // Each layer: { name, cols, rows, x, scale }
  const LAYERS = [
    { name: 'input',  cols: 14, rows: 14, x: -9,  scale: 1.0 },  // 14×14 = 196 dots (sampled from 28×28)
    { name: 'conv1',  cols:  7, rows:  7, x: -3,  scale: 1.0 },  // conv feature maps
    { name: 'conv2',  cols:  5, rows:  5, x:  2,  scale: 1.0 },  // deeper features
    { name: 'dense',  cols:  4, rows:  9, x:  6.5,scale: 1.0 },  // dense 36 output
  ];

  // Connections between consecutive layers (sparse for performance)
  const MAX_CONNECTIONS = 280;

  // ─── Init ─────────────────────────────────────────────────────
  function init() {
    const container = document.getElementById('three-container');
    if (!container) return;

    const W = container.clientWidth;
    const H = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x04040f);
    scene.fog = new THREE.FogExp2(0x04040f, 0.025);

    // Camera
    camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 200);
    camera.position.set(0, 3, 18);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    // Orbit controls
    if (THREE.OrbitControls) {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping    = true;
      controls.dampingFactor    = 0.08;
      controls.enablePan        = false;
      controls.minDistance      = 6;
      controls.maxDistance      = 35;
      controls.autoRotate       = autoRotate;
      controls.autoRotateSpeed  = 0.4;
      controls.target.set(0, 0, 0);
    }

    // Ambient + point lights
    scene.add(new THREE.AmbientLight(0x112233, 1.2));
    const ptLight = new THREE.PointLight(0x00e5ff, 1.5, 30);
    ptLight.position.set(0, 5, 5);
    scene.add(ptLight);

    // Build neural network geometry
    buildLayers();
    buildConnections();

    // Resize handler
    window.addEventListener('resize', onResize);

    // Start render loop
    animate();

    // Expose public API
    window.NeuralViz = {
      triggerInference,
      setInputPixels,
      toggleAutoRotate,
      resetCamera,
    };
  }

  // ─── Build Layers ─────────────────────────────────────────────
  function buildLayers() {
    const dotGeo = new THREE.SphereGeometry(0.08, 8, 8);

    LAYERS.forEach((layerDef, li) => {
      const group   = new THREE.Group();
      const neurons = [];

      const totalW = (layerDef.cols - 1) * 0.55;
      const totalH = (layerDef.rows - 1) * 0.55;

      for (let r = 0; r < layerDef.rows; r++) {
        for (let c = 0; c < layerDef.cols; c++) {
          const mat = new THREE.MeshStandardMaterial({
            color:     COL_CYAN,
            emissive:  COL_CYAN,
            emissiveIntensity: 0.35,
            roughness: 0.2,
            metalness: 0.1,
          });

          const mesh = new THREE.Mesh(dotGeo, mat);
          mesh.position.set(
            layerDef.x,
            (r / (layerDef.rows - 1)) * totalH - totalH / 2,
            (c / (layerDef.cols - 1)) * totalW - totalW / 2
          );
          mesh.userData = { baseIntensity: 0.35, li, r, c };
          group.add(mesh);
          neurons.push(mesh);
        }
      }

      // Slight tilt for 3D depth effect (like reference images)
      group.rotation.y = li === 0 ? 0.3 : li === 1 ? 0.15 : 0.05;

      scene.add(group);
      layerGroups.push(group);
      layerNeurons.push(neurons);
    });
  }

  // ─── Build Connections ────────────────────────────────────────
  function buildConnections() {
    for (let li = 0; li < LAYERS.length - 1; li++) {
      const srcNeurons = layerNeurons[li];
      const dstNeurons = layerNeurons[li + 1];
      const connGroup  = new THREE.Group();

      const count = Math.min(MAX_CONNECTIONS, srcNeurons.length * 2);
      const lines = [];

      for (let i = 0; i < count; i++) {
        const src = srcNeurons[Math.floor(Math.random() * srcNeurons.length)];
        const dst = dstNeurons[Math.floor(Math.random() * dstNeurons.length)];

        const points = [
          src.getWorldPosition(new THREE.Vector3()),
          dst.getWorldPosition(new THREE.Vector3()),
        ];

        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({
          color:       COL_PURPLE,
          opacity:     0.12,
          transparent: true,
        });

        const line = new THREE.Line(geo, mat);
        line.userData = { src, dst, baseOpacity: 0.12, phase: Math.random() * Math.PI * 2 };
        connGroup.add(line);
        lines.push(line);
      }

      scene.add(connGroup);
      connectionGroups.push({ group: connGroup, lines });
    }
  }

  // ─── Animate ──────────────────────────────────────────────────
  let clock = new THREE.Clock();

  function animate() {
    animFrameId = requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    // Orbit controls update
    if (controls) controls.update();

    // Idle neuron pulsing
    layerNeurons.forEach((neurons, li) => {
      neurons.forEach((n, ni) => {
        if (!inferenceActive) {
          const pulse = 0.2 + 0.15 * Math.sin(t * 1.2 + ni * 0.3 + li * 1.5);
          n.material.emissiveIntensity = pulse;
          n.scale.setScalar(0.8 + 0.2 * Math.sin(t * 0.8 + ni * 0.2));
        }
      });
    });

    // Idle connection shimmer
    connectionGroups.forEach(({ lines }, ci) => {
      lines.forEach((line, li) => {
        if (!inferenceActive) {
          const shimmer = 0.06 + 0.08 * Math.sin(t * 0.7 + line.userData.phase);
          line.material.opacity = shimmer;
        }
      });
    });

    // Inference wave animation
    if (inferenceActive) {
      updateInferenceAnimation(t);
    }

    renderer.render(scene, camera);
  }

  // ─── Inference Animation ──────────────────────────────────────
  let inferenceStartTime = 0;
  const INFERENCE_DURATION = 2.2; // seconds

  function updateInferenceAnimation(t) {
    const elapsed  = t - inferenceStartTime;
    const progress = Math.min(elapsed / INFERENCE_DURATION, 1.0);
    const waveFront = progress * (LAYERS.length + 0.5);

    // Animate layers
    layerNeurons.forEach((neurons, li) => {
      const layerProgress = waveFront - li;
      const inWave        = layerProgress > 0 && layerProgress < 1.5;

      neurons.forEach((n, ni) => {
        if (inWave) {
          // Activate based on score if output layer
          const intensity = li === LAYERS.length - 1
            ? (inferenceScores[ni] || 0) * 2.5
            : 0.6 + 0.4 * Math.sin(t * 8 + ni * 0.5);

          n.material.emissive    = li === LAYERS.length - 1
            ? lerpColor(COL_DIM, COL_GREEN, inferenceScores[ni] || 0)
            : COL_CYAN;
          n.material.emissiveIntensity = intensity;
          n.scale.setScalar(1.0 + intensity * 0.5);
        } else {
          n.material.emissive          = COL_CYAN;
          n.material.emissiveIntensity = 0.2;
          n.scale.setScalar(0.9);
        }
      });
    });

    // Animate connections
    connectionGroups.forEach(({ lines }, ci) => {
      const connProgress = waveFront - ci - 0.5;
      const active       = connProgress > 0 && connProgress < 1.2;

      lines.forEach((line) => {
        if (active) {
          const weight    = Math.random() > 0.5 ? 1 : -1;
          line.material.color   = weight > 0 ? COL_GREEN : COL_RED;
          line.material.opacity = 0.55 + 0.35 * Math.sin(t * 6 + line.userData.phase);
        } else {
          line.material.color   = COL_PURPLE;
          line.material.opacity = 0.08;
        }
      });
    });

    if (progress >= 1.0) {
      inferenceActive = false;
      // Fade back to idle
      setTimeout(resetNeuronColors, 500);
    }
  }

  function resetNeuronColors() {
    layerNeurons.forEach(neurons => {
      neurons.forEach(n => {
        n.material.emissive          = COL_CYAN;
        n.material.emissiveIntensity = 0.35;
        n.scale.setScalar(1.0);
      });
    });
    connectionGroups.forEach(({ lines }) => {
      lines.forEach(line => {
        line.material.color   = COL_PURPLE;
        line.material.opacity = 0.12;
      });
    });
  }

  // ─── Public: trigger inference animation ──────────────────────
  function triggerInference(scores) {
    inferenceScores  = scores || [];
    inferenceActive  = true;
    inferenceStartTime = clock.getElapsedTime();
  }

  // ─── Public: set input pixels ─────────────────────────────────
  function setInputPixels(pixelData) {
    // pixelData: flat array of 0-1 values (28x28 = 784, sampled to 196)
    const neurons = layerNeurons[0];
    if (!neurons || !pixelData) return;

    const step = Math.floor(pixelData.length / neurons.length);
    neurons.forEach((n, i) => {
      const val = pixelData[i * step] || 0;
      n.material.emissiveIntensity = 0.1 + val * 1.2;
      n.material.emissive          = val > 0.3 ? COL_CYAN : COL_DIM;
      n.scale.setScalar(0.7 + val * 0.6);
    });
  }

  // ─── Public: toggle auto rotate ──────────────────────────────
  function toggleAutoRotate() {
    autoRotate = !autoRotate;
    if (controls) controls.autoRotate = autoRotate;
    return autoRotate;
  }

  // ─── Public: reset camera ────────────────────────────────────
  function resetCamera() {
    camera.position.set(0, 3, 18);
    if (controls) { controls.target.set(0, 0, 0); controls.update(); }
  }

  // ─── Resize ──────────────────────────────────────────────────
  function onResize() {
    const container = document.getElementById('three-container');
    if (!container) return;
    const W = container.clientWidth;
    const H = container.clientHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  }

  // ─── Helper: lerp colors ─────────────────────────────────────
  function lerpColor(a, b, t) {
    return new THREE.Color(
      a.r + (b.r - a.r) * t,
      a.g + (b.g - a.g) * t,
      a.b + (b.b - a.b) * t,
    );
  }

  // ─── Background particle canvas ───────────────────────────────
  function initBgParticles() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    const ctx    = canvas.getContext('2d');
    let W, H, particles;

    function resize() {
      W = canvas.width  = window.innerWidth;
      H = canvas.height = window.innerHeight;
    }

    function makeParticles() {
      particles = Array.from({ length: 80 }, () => ({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 1.5 + 0.3,
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
        a: Math.random(),
      }));
    }

    function drawBg() {
      ctx.clearRect(0, 0, W, H);
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = W;
        if (p.x > W) p.x = 0;
        if (p.y < 0) p.y = H;
        if (p.y > H) p.y = 0;
        p.a += 0.005;
        const alpha = 0.15 + 0.1 * Math.sin(p.a);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0,229,255,${alpha})`;
        ctx.fill();
      });
      requestAnimationFrame(drawBg);
    }

    resize();
    makeParticles();
    drawBg();
    window.addEventListener('resize', () => { resize(); makeParticles(); });
  }

  // ─── Boot ────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    initBgParticles();
    init();
  });

})();

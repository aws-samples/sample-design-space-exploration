import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

/**
 * Professional 3D STL viewer — viewstl.com style.
 * Model fills the viewport, centered, clean dark background,
 * smooth orbit/pan/zoom, model info overlay.
 */
const STLViewer = ({ url }) => {
  const mountRef = useRef(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [modelInfo, setModelInfo] = useState(null);
  const [wireframe, setWireframe] = useState(false);
  const meshRef = useRef(null);
  const edgesRef = useRef(null);

  useEffect(() => {
    if (!url || !mountRef.current) return;

    const container = mountRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    // Scene — clean gradient background like viewstl
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // Camera — use 35° FOV for less distortion, more "product shot" feel
    const camera = new THREE.PerspectiveCamera(35, width / height, 0.001, 100000);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    // Lighting — soft studio setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    keyLight.position.set(1, 1.5, 1);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x8ec8ff, 0.7);
    fillLight.position.set(-1, 0.5, -0.5);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xfff0dd, 0.4);
    rimLight.position.set(0, -0.5, -1);
    scene.add(rimLight);

    const hemiLight = new THREE.HemisphereLight(0xc8d8ff, 0x303040, 0.6);
    scene.add(hemiLight);

    // Orbit state — all relative to bounding sphere radius
    let orbitCenter = new THREE.Vector3(0, 0, 0);
    let sphereRadius = 1;
    let isDragging = false;
    let isPanning = false;
    let prevMouse = { x: 0, y: 0 };
    let theta = 0.8;       // horizontal angle
    let phi = 0.45;        // vertical angle (slightly above)
    let distFactor = 2.8;  // camera distance as multiple of bounding sphere radius
    let targetTheta = 0.8;
    let targetPhi = 0.45;
    let targetDist = 2.8;
    let targetCenter = new THREE.Vector3(0, 0, 0);

    // Load STL
    fetch(url)
      .then(res => res.arrayBuffer())
      .then(buffer => {
        const geometry = parseSTL(buffer);
        if (!geometry) { setError('Failed to parse STL file'); setLoading(false); return; }

        geometry.computeVertexNormals();

        // Material — viewstl.com uses a clean light gray with slight sheen
        const material = new THREE.MeshStandardMaterial({
          color: 0xc0c0c0,
          metalness: 0.15,
          roughness: 0.5,
          side: THREE.DoubleSide,
          flatShading: false,
          envMapIntensity: 0.8,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        meshRef.current = mesh;

        // Compute bounding sphere — this is the key to proper framing
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
        const bSphere = geometry.boundingSphere;
        const bBox = geometry.boundingBox;
        const center = bSphere.center.clone();
        sphereRadius = bSphere.radius;

        // DON'T scale the mesh — keep original units. Instead, position camera relative to bounding sphere.
        scene.add(mesh);

        // Edge wireframe overlay
        const edgesGeo = new THREE.EdgesGeometry(geometry, 30);
        const edgesMat = new THREE.LineBasicMaterial({ color: 0x666666, transparent: true, opacity: 0.2 });
        const edges = new THREE.LineSegments(edgesGeo, edgesMat);
        edges.visible = false;
        edgesRef.current = edges;
        scene.add(edges);

        // Subtle ground plane at the bottom of the model
        const bSize = new THREE.Vector3();
        bBox.getSize(bSize);
        const groundSize = Math.max(bSize.x, bSize.z) * 3;
        const groundGeo = new THREE.PlaneGeometry(groundSize, groundSize);
        const groundMat = new THREE.ShadowMaterial({ opacity: 0.15 });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = bBox.min.y;
        ground.receiveShadow = true;
        scene.add(ground);

        // Subtle grid — scaled to model, not dominating
        const gridSize = Math.max(bSize.x, bSize.z) * 2;
        const gridDivisions = 20;
        const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0x333355, 0x252545);
        gridHelper.material.opacity = 0.25;
        gridHelper.material.transparent = true;
        gridHelper.position.y = bBox.min.y;
        scene.add(gridHelper);

        // Position lights relative to model size
        const lightDist = sphereRadius * 3;
        keyLight.position.set(lightDist, lightDist * 1.5, lightDist);
        keyLight.shadow.camera.near = sphereRadius * 0.1;
        keyLight.shadow.camera.far = sphereRadius * 10;
        keyLight.shadow.camera.left = -sphereRadius * 2;
        keyLight.shadow.camera.right = sphereRadius * 2;
        keyLight.shadow.camera.top = sphereRadius * 2;
        keyLight.shadow.camera.bottom = -sphereRadius * 2;
        fillLight.position.set(-lightDist, lightDist * 0.5, -lightDist * 0.5);
        rimLight.position.set(0, -lightDist * 0.3, -lightDist);

        // Camera near/far relative to model
        camera.near = sphereRadius * 0.01;
        camera.far = sphereRadius * 100;
        camera.updateProjectionMatrix();

        // Set orbit center to model center
        orbitCenter.copy(center);
        targetCenter.copy(center);

        // Compute ideal camera distance to fill ~75% of viewport
        // Using: distance = radius / sin(fov/2) gives exact fit
        // Multiply by 1.15 for a bit of padding (viewstl style)
        const fovRad = camera.fov * Math.PI / 180;
        const aspect = camera.aspect;
        const hFov = 2 * Math.atan(Math.tan(fovRad / 2) * aspect);
        const fitDist = sphereRadius / Math.sin(Math.min(fovRad, hFov) / 2);
        distFactor = fitDist / sphereRadius * 1.15;
        targetDist = distFactor;

        // Initial camera angle — slightly above and to the side (3/4 view like viewstl)
        targetTheta = Math.PI * 0.75;
        targetPhi = Math.PI * 0.15;
        theta = targetTheta;
        phi = targetPhi;

        // Model info
        const triCount = geometry.attributes.position.count / 3;
        setModelInfo({
          triangles: triCount.toLocaleString(),
          vertices: geometry.attributes.position.count.toLocaleString(),
          sizeX: bSize.x.toFixed(2),
          sizeY: bSize.y.toFixed(2),
          sizeZ: bSize.z.toFixed(2),
          fileSize: (buffer.byteLength / 1024).toFixed(1),
        });

        setLoading(false);
      })
      .catch(err => { setError('Failed to load STL: ' + err.message); setLoading(false); });

    // --- Mouse / touch controls ---
    const onMouseDown = (e) => {
      if (e.button === 2 || e.button === 1) { isPanning = true; }
      else { isDragging = true; }
      prevMouse = { x: e.clientX, y: e.clientY };
      e.preventDefault();
    };
    const onMouseMove = (e) => {
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      if (isDragging) {
        targetTheta -= dx * 0.005;
        targetPhi += dy * 0.005;
        targetPhi = Math.max(0.05, Math.min(Math.PI - 0.05, targetPhi));
      }
      if (isPanning) {
        // Pan in camera-local space, scaled to model size
        const panScale = sphereRadius * targetDist * 0.001;
        const right = new THREE.Vector3();
        const up = new THREE.Vector3(0, 1, 0);
        const camDir = new THREE.Vector3();
        camera.getWorldDirection(camDir);
        right.crossVectors(camDir, up).normalize();
        const camUp = new THREE.Vector3();
        camUp.crossVectors(right, camDir).normalize();
        targetCenter.addScaledVector(right, -dx * panScale);
        targetCenter.addScaledVector(camUp, dy * panScale);
      }
      prevMouse = { x: e.clientX, y: e.clientY };
    };
    const onMouseUp = () => { isDragging = false; isPanning = false; };
    const onContextMenu = (e) => e.preventDefault();

    const onWheel = (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.1 : 0.9;
      targetDist = Math.max(1.2, Math.min(20, targetDist * factor));
    };

    // Touch support
    let lastTouchDist = 0;
    const onTouchStart = (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        isDragging = false;
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastTouchDist = Math.sqrt(dx * dx + dy * dy);
      }
    };
    const onTouchMove = (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && isDragging) {
        const dx = e.touches[0].clientX - prevMouse.x;
        const dy = e.touches[0].clientY - prevMouse.y;
        targetTheta -= dx * 0.005;
        targetPhi += dy * 0.005;
        targetPhi = Math.max(0.05, Math.min(Math.PI - 0.05, targetPhi));
        prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (lastTouchDist > 0) {
          const factor = lastTouchDist / dist;
          targetDist = Math.max(1.2, Math.min(20, targetDist * factor));
        }
        lastTouchDist = dist;
      }
    };
    const onTouchEnd = () => { isDragging = false; lastTouchDist = 0; };

    container.addEventListener('mousedown', onMouseDown);
    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('mouseup', onMouseUp);
    container.addEventListener('mouseleave', onMouseUp);
    container.addEventListener('contextmenu', onContextMenu);
    container.addEventListener('wheel', onWheel, { passive: false });
    container.addEventListener('touchstart', onTouchStart, { passive: false });
    container.addEventListener('touchmove', onTouchMove, { passive: false });
    container.addEventListener('touchend', onTouchEnd);

    // Animation loop — spherical coordinates orbit
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      // Smooth interpolation
      const lerp = 0.1;
      theta += (targetTheta - theta) * lerp;
      phi += (targetPhi - phi) * lerp;
      distFactor += (targetDist - distFactor) * lerp;
      orbitCenter.lerp(targetCenter, lerp);

      // Spherical to Cartesian — phi is polar angle from top
      const r = sphereRadius * distFactor;
      camera.position.set(
        orbitCenter.x + r * Math.sin(phi) * Math.cos(theta),
        orbitCenter.y + r * Math.cos(phi),
        orbitCenter.z + r * Math.sin(phi) * Math.sin(theta)
      );
      camera.lookAt(orbitCenter);

      renderer.render(scene, camera);
    };
    animate();

    // Resize
    const onResize = () => {
      const w = container.clientWidth || 800;
      const h = container.clientHeight || 600;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', onResize);
      container.removeEventListener('mousedown', onMouseDown);
      container.removeEventListener('mousemove', onMouseMove);
      container.removeEventListener('mouseup', onMouseUp);
      container.removeEventListener('mouseleave', onMouseUp);
      container.removeEventListener('contextmenu', onContextMenu);
      container.removeEventListener('wheel', onWheel);
      container.removeEventListener('touchstart', onTouchStart);
      container.removeEventListener('touchmove', onTouchMove);
      container.removeEventListener('touchend', onTouchEnd);
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, [url]);

  // Toggle wireframe
  useEffect(() => {
    if (edgesRef.current) edgesRef.current.visible = wireframe;
  }, [wireframe]);

  if (error) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        backgroundColor: '#1a1a2e', color: '#ff6b6b', padding: 20 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>⚠️</div>
          <p style={{ fontSize: 14 }}>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, position: 'relative', backgroundColor: '#1a1a2e' }}>
      {loading && (
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          backgroundColor: '#1a1a2e', zIndex: 10
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              width: 48, height: 48, border: '3px solid #333',
              borderTop: '3px solid #FF9900', borderRadius: '50%',
              animation: 'spin 1s linear infinite', margin: '0 auto 16px'
            }} />
            <p style={{ color: '#888', fontSize: 14 }}>Loading 3D model...</p>
          </div>
        </div>
      )}

      <div ref={mountRef} style={{ width: '100%', height: '100%', minHeight: 500 }} />

      {/* Toolbar */}
      <div style={{ position: 'absolute', top: 12, right: 12, display: 'flex', gap: 6, zIndex: 5 }}>
        <button onClick={() => setWireframe(w => !w)}
          style={{
            padding: '6px 14px', borderRadius: 6, border: 'none',
            backgroundColor: wireframe ? '#FF9900' : 'rgba(255,255,255,0.1)',
            color: wireframe ? '#000' : '#aaa', fontSize: 12, fontWeight: 600,
            cursor: 'pointer', transition: 'all 0.2s ease'
          }}>
          {wireframe ? '◼ Solid+Edges' : '◻ Edges'}
        </button>
      </div>

      {/* Model info */}
      {modelInfo && (
        <div style={{
          position: 'absolute', bottom: 12, left: 12,
          backgroundColor: 'rgba(0,0,0,0.65)', padding: '10px 14px',
          borderRadius: 8, fontSize: 11, color: '#999', lineHeight: 1.7,
          backdropFilter: 'blur(8px)', zIndex: 5, fontFamily: 'monospace'
        }}>
          <div style={{ color: '#ccc', fontWeight: 700, marginBottom: 4, fontSize: 12 }}>Model Info</div>
          <div>Triangles: <span style={{ color: '#FF9900' }}>{modelInfo.triangles}</span></div>
          <div>Vertices: <span style={{ color: '#FF9900' }}>{modelInfo.vertices}</span></div>
          <div>Size: <span style={{ color: '#FF9900' }}>{modelInfo.sizeX} × {modelInfo.sizeY} × {modelInfo.sizeZ}</span></div>
          <div>File: <span style={{ color: '#FF9900' }}>{modelInfo.fileSize} KB</span></div>
        </div>
      )}

      {/* Controls hint */}
      <div style={{
        position: 'absolute', bottom: 12, right: 12,
        backgroundColor: 'rgba(0,0,0,0.5)', padding: '8px 12px',
        borderRadius: 8, fontSize: 11, color: '#777',
        backdropFilter: 'blur(8px)', zIndex: 5, lineHeight: 1.6
      }}>
        <div>🖱️ Left drag: Rotate</div>
        <div>🖱️ Right drag: Pan</div>
        <div>🖱️ Scroll: Zoom</div>
      </div>
    </div>
  );
};

/**
 * Parse binary or ASCII STL into a THREE.BufferGeometry.
 */
function parseSTL(buffer) {
  const view = new DataView(buffer);
  const decoder = new TextDecoder();
  const header = decoder.decode(new Uint8Array(buffer, 0, Math.min(80, buffer.byteLength)));

  if (header.startsWith('solid') && buffer.byteLength > 84) {
    const triCount = view.getUint32(80, true);
    const expectedSize = 84 + triCount * 50;
    if (Math.abs(expectedSize - buffer.byteLength) > 10) {
      return parseASCIISTL(decoder.decode(new Uint8Array(buffer)));
    }
  }

  const triangleCount = view.getUint32(80, true);
  if (triangleCount === 0) return null;

  const positions = new Float32Array(triangleCount * 9);
  const normals = new Float32Array(triangleCount * 9);

  let offset = 84;
  for (let i = 0; i < triangleCount; i++) {
    const nx = view.getFloat32(offset, true); offset += 4;
    const ny = view.getFloat32(offset, true); offset += 4;
    const nz = view.getFloat32(offset, true); offset += 4;

    for (let v = 0; v < 3; v++) {
      const idx = i * 9 + v * 3;
      positions[idx]     = view.getFloat32(offset, true); offset += 4;
      positions[idx + 1] = view.getFloat32(offset, true); offset += 4;
      positions[idx + 2] = view.getFloat32(offset, true); offset += 4;
      normals[idx]     = nx;
      normals[idx + 1] = ny;
      normals[idx + 2] = nz;
    }
    offset += 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return geometry;
}

function parseASCIISTL(text) {
  const positions = [];
  const normals = [];
  const vertexRegex = /vertex\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)/g;
  const normalRegex = /facet\s+normal\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)/g;

  let nMatch;
  const normalList = [];
  while ((nMatch = normalRegex.exec(text)) !== null) {
    normalList.push([parseFloat(nMatch[1]), parseFloat(nMatch[2]), parseFloat(nMatch[3])]);
  }

  let vMatch;
  let vIdx = 0;
  while ((vMatch = vertexRegex.exec(text)) !== null) {
    positions.push(parseFloat(vMatch[1]), parseFloat(vMatch[2]), parseFloat(vMatch[3]));
    const nIdx = Math.floor(vIdx / 3);
    if (nIdx < normalList.length) {
      normals.push(...normalList[nIdx]);
    } else {
      normals.push(0, 0, 1);
    }
    vIdx++;
  }

  if (positions.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
  return geometry;
}

export default STLViewer;

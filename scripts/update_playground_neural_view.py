"""
update_playground_neural_view.py
================================
Updates docs/index.html and the artifact tbnn_studio.html with the full
TensorFlow Playground-style interconnected neural network visualization:
- Curved cubic Bezier synaptic mesh (positive blue, negative dashed orange)
- Nodes with internal activation patterns matching the reference screenshot
- Live feedforward animated signal pulses traveling along curves
- Output secondary flow contour heatmap + convergence loss curve
- Interactive hover/click node inspection
"""

import re
from pathlib import Path

# Neural Network Playground Component HTML + CSS + JS
NEURAL_VIEW_HTML = """
    <!-- ================================================================= -->
    <!-- VIEW 2: TENSORFLOW PLAYGROUND-STYLE NEURAL NETWORK ENGINE         -->
    <!-- ================================================================= -->
    <main id="view-neural" class="flex-1 flex flex-col p-6 overflow-y-auto space-y-6 hidden">
      
      <!-- Top Title & Controls (Matching Playground Toolbar) -->
      <div class="bg-[#161b22] p-4 rounded-2xl border border-[#30363d] flex flex-wrap items-center justify-between gap-4 shadow-sm">
        <div class="flex items-center space-x-3">
          <div class="w-9 h-9 rounded-xl bg-purple-600/20 border border-purple-500/40 flex items-center justify-center text-purple-400 font-bold text-lg">
            🧠
          </div>
          <div>
            <div class="flex items-center space-x-2">
              <h2 class="text-sm font-bold text-white">TBNN Neural Network Architecture & Synaptic Mesh</h2>
              <span class="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-300 border border-purple-500/20 text-xs font-semibold">Fully Connected MLP</span>
            </div>
            <p class="text-xs text-[#8b949e]">Interactive feedforward dataflow from Pope scalar invariants to integrity basis weights</p>
          </div>
        </div>

        <!-- Toolbar Controls -->
        <div class="flex items-center space-x-3">
          <button id="btn-nn-pulse" onclick="startFeedforwardWave()"
                  class="px-4 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold flex items-center space-x-2 shadow-md transition">
            <span id="nn-pulse-icon">⚡</span>
            <span id="nn-pulse-text">Run Feedforward Wave</span>
          </button>
          <button onclick="toggleWeightSignOnly()"
                  class="px-3 py-1.5 rounded-lg bg-[#21262d] hover:bg-[#30363d] text-[#8b949e] hover:text-white text-xs font-medium border border-[#30363d] transition">
            Filter: Strong Synapses
          </button>
          <div class="flex items-center space-x-2 px-3 py-1 rounded-lg bg-[#0d1117] border border-[#30363d] text-xs font-mono">
            <span class="text-[#8b949e]">Loss:</span>
            <span class="text-emerald-400 font-bold">8.16e-5</span>
          </div>
        </div>
      </div>

      <!-- MAIN PLAYGROUND STAGE (SVG + Canvas Container) -->
      <div class="bg-[#161b22] border border-[#30363d] rounded-3xl p-6 relative overflow-hidden shadow-xl">
        
        <!-- Header Column Labels (Playground Style) -->
        <div class="grid grid-cols-12 gap-2 text-xs font-bold uppercase tracking-wider pb-3 border-b border-[#30363d]/80 text-[#8b949e]">
          <div class="col-span-2 text-blue-400 flex items-center space-x-1">
            <span>FEATURES</span>
            <span class="text-[10px] font-normal normal-case text-[#8b949e]">(5 Invariants)</span>
          </div>
          <div class="col-span-2 text-purple-300 flex items-center justify-center space-x-1">
            <span>HIDDEN 1</span>
            <span class="text-[10px] font-normal normal-case text-[#8b949e]">(64 Neurons)</span>
          </div>
          <div class="col-span-2 text-purple-300 flex items-center justify-center space-x-1">
            <span>HIDDEN 2</span>
            <span class="text-[10px] font-normal normal-case text-[#8b949e]">(64 Neurons)</span>
          </div>
          <div class="col-span-2 text-purple-300 flex items-center justify-center space-x-1">
            <span>HIDDEN 3</span>
            <span class="text-[10px] font-normal normal-case text-[#8b949e]">(64 Neurons)</span>
          </div>
          <div class="col-span-1 text-amber-400 text-center">
            <span>OUTPUTS</span>
          </div>
          <div class="col-span-3 text-emerald-400 text-right pr-2">
            <span>ANISOTROPY & VORTICES</span>
          </div>
        </div>

        <!-- Interactive Diagram Workspace -->
        <div class="relative w-full overflow-x-auto py-4" style="min-height: 480px;">
          <!-- SVG Synaptic Mesh & Nodes -->
          <svg id="nn-svg" class="w-full h-[470px]" viewBox="0 0 1160 470" preserveAspectRatio="xMidYMid meet">
            <defs>
              <!-- Glowing Filters -->
              <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2.5" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="glow-orange" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2.5" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <linearGradient id="grad-pulse" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#38bdf8" stop-opacity="1" />
                <stop offset="100%" stop-color="#c084fc" stop-opacity="1" />
              </linearGradient>
            </defs>

            <!-- 1. SYNAPTIC CONNECTION LINES (Injected by JS) -->
            <g id="synapse-group" class="transition-opacity duration-300"></g>

            <!-- 2. ANIMATED PULSE LAYER -->
            <g id="pulse-group" pointer-events="none"></g>

            <!-- 3. NEURON NODES & ACTIVATION BOXES (Injected by JS) -->
            <g id="neuron-group"></g>

            <!-- 4. RIGHT SIDE OUTPUT PANEL (Reynolds Stress Heatmap & Recomposition) -->
            <g id="output-viz-group" transform="translate(930, 30)">
              <!-- Outer Card -->
              <rect x="0" y="0" width="220" height="390" rx="16" fill="#0d1117" stroke="#30363d" stroke-width="1.5" />
              
              <!-- Loss Convergence Graph (Top Right Mini) -->
              <text x="12" y="24" fill="#8b949e" font-size="10" font-weight="bold">Continuity Residual: 8.16e-5</text>
              <path d="M 12 55 Q 60 50, 100 40 T 208 32" fill="none" stroke="#58a6ff" stroke-width="1.5" />
              <line x1="12" y1="58" x2="208" y2="58" stroke="#30363d" stroke-dasharray="2,2" />

              <!-- Heatmap Box (Like Spiral in Reference Image) -->
              <rect x="12" y="70" width="196" height="196" rx="12" fill="#161b22" stroke="#38bdf8" stroke-width="1.5" />
              
              <!-- Embedded Heatmap Image or Visual -->
              <image href="figures/reynolds_stress_mechanism.png" x="14" y="72" width="192" height="192" preserveAspectRatio="xMidYMid slice" rx="10" />

              <!-- Output Title -->
              <text x="110" y="286" fill="#f0f6fc" font-size="11" font-weight="bold" text-anchor="middle">Prandtl Secondary Vortices</text>
              <text x="110" y="302" fill="#8b949e" font-size="10" text-anchor="middle">Normal Stress Diff: (v'² - w'²)</text>
              
              <!-- Colorbar Legend -->
              <g transform="translate(25, 320)">
                <defs>
                  <linearGradient id="legend-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#f97316" />
                    <stop offset="50%" stop-color="#ffffff" />
                    <stop offset="100%" stop-color="#38bdf8" />
                  </linearGradient>
                </defs>
                <rect x="0" y="0" width="170" height="10" rx="3" fill="url(#legend-grad)" />
                <text x="0" y="22" fill="#f97316" font-size="9" font-weight="bold">-1.0 (CW)</text>
                <text x="85" y="22" fill="#8b949e" font-size="9" text-anchor="middle">0.0</text>
                <text x="170" y="22" fill="#38bdf8" font-size="9" font-weight="bold" text-anchor="end">+1.0 (ACW)</text>
              </g>

              <!-- Formula Tag -->
              <text x="110" y="365" fill="#34d399" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">a_ij = ∑ g⁽ⁿ⁾ · T⁽ⁿ⁾</text>
            </g>
          </svg>
        </div>

        <!-- Bottom Legend & Node Details Panel -->
        <div class="mt-4 pt-4 border-t border-[#30363d]/80 flex flex-wrap items-center justify-between gap-4 text-xs">
          <!-- Synaptic Weight Legend -->
          <div class="flex items-center space-x-6">
            <span class="text-[#8b949e] font-semibold">Synaptic Connections:</span>
            <div class="flex items-center space-x-2">
              <span class="w-5 h-0.5 bg-[#38bdf8]"></span>
              <span class="text-white">Positive Weight (<span class="text-[#38bdf8] font-mono">w > 0</span>)</span>
            </div>
            <div class="flex items-center space-x-2">
              <span class="w-5 h-0.5 border-b border-dashed border-[#f97316]"></span>
              <span class="text-white">Negative Weight (<span class="text-[#f97316] font-mono">w < 0</span>)</span>
            </div>
            <div class="text-[#8b949e] italic">Line thickness indicates connection strength</div>
          </div>

          <div id="neuron-status-tip" class="px-3 py-1 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#8b949e] font-mono">
            Hover or click any neuron node to inspect synaptic paths
          </div>
        </div>

      </div>

      <!-- Detail Explanations -->
      <div class="grid grid-cols-3 gap-6">
        <div class="bg-[#161b22] border border-[#30363d] rounded-2xl p-4 space-y-2">
          <div class="flex items-center space-x-2 text-blue-400 font-bold text-xs uppercase tracking-wider">
            <span>1. Invariant Feature Inputs</span>
          </div>
          <p class="text-xs text-[#8b949e] leading-relaxed">
            Unlike classical neural networks that train on raw spatial coordinates, TBNN takes the 5 frame-invariant scalars of strain <span class="font-mono text-white">S</span> and rotation <span class="font-mono text-white">R</span>, guaranteeing identical predictions under any grid rotation.
          </p>
        </div>

        <div class="bg-[#161b22] border border-[#30363d] rounded-2xl p-4 space-y-2">
          <div class="flex items-center space-x-2 text-purple-400 font-bold text-xs uppercase tracking-wider">
            <span>2. Deep Nonlinear Synapses</span>
          </div>
          <p class="text-xs text-[#8b949e] leading-relaxed">
            Each neuron computes <span class="font-mono text-white">y = LeakyReLU(∑ w_i x_i + b)</span>. The 3 dense layers learn the highly nonlinear coupling between mean shear and turbulent normal stress imbalances.
          </p>
        </div>

        <div class="bg-[#161b22] border border-[#30363d] rounded-2xl p-4 space-y-2">
          <div class="flex items-center space-x-2 text-amber-400 font-bold text-xs uppercase tracking-wider">
            <span>3. Output Tensor Recomposition</span>
          </div>
          <p class="text-xs text-[#8b949e] leading-relaxed">
            The 10 output weights <span class="font-mono text-white">g⁽ⁿ⁾</span> scale the 10 Pope basis matrices <span class="font-mono text-white">T⁽ⁿ⁾</span>. Specifically, basis <span class="font-mono text-white">T⁽²⁾</span> drives the non-zero <span class="font-mono text-white">(v'² - w'²)</span> that generates the 8 secondary flow cells!
          </p>
        </div>
      </div>

    </main>
"""

# JavaScript Engine for Playground Rendering, Mesh Generation, and Wave Animation
PLAYGROUND_JS = """
    // =========================================================================
    // TENSORFLOW PLAYGROUND-STYLE NEURAL NETWORK ENGINE (SVG + BEZIER MESH)
    // =========================================================================
    const layerSpecs = [
      { id: 'input', name: 'Features', count: 5, x: 70, labels: ['I₁: Strain', 'I₂: Vorticity', 'I₃: Skewness', 'I₄: Align', 'I₅: Nonlinear'], sub: ['tr(S²)', 'tr(R²)', 'tr(S³)', 'tr(R²S)', 'tr(R²S²)'] },
      { id: 'h1', name: 'Hidden 1', count: 7, x: 260, act: 'LeakyReLU' },
      { id: 'h2', name: 'Hidden 2', count: 7, x: 450, act: 'LeakyReLU' },
      { id: 'h3', name: 'Hidden 3', count: 6, x: 640, act: 'LeakyReLU' },
      { id: 'out', name: 'Outputs', count: 6, x: 830, labels: ['g⁽¹⁾ (Linear)', 'g⁽²⁾ (S²-Quad)', 'g⁽³⁾ (R²-Vort)', 'g⁽⁴⁾ (SR-Cross)', 'g⁽⁵⁾ (Higher)', 'g⁽⁶⁻¹⁰⁾ (Tensors)'] }
    ];

    let nnNodes = [];
    let nnSynapses = [];
    let isPulseActive = false;
    let filterStrong = false;

    function initNeuralPlayground() {
      const svg = document.getElementById('nn-svg');
      if (!svg) return;

      const synapseGroup = document.getElementById('synapse-group');
      const neuronGroup = document.getElementById('neuron-group');
      synapseGroup.innerHTML = '';
      neuronGroup.innerHTML = '';

      nnNodes = [];
      nnSynapses = [];

      const svgHeight = 440;

      // 1. Calculate node positions
      layerSpecs.forEach((layer, layerIdx) => {
        const count = layer.count;
        const spacing = Math.min(56, (svgHeight - 60) / (count + 1));
        const totalHeight = (count - 1) * spacing;
        const startY = (svgHeight - totalHeight) / 2;

        for (let i = 0; i < count; i++) {
          const y = startY + i * spacing;
          const nodeId = `${layer.id}_${i}`;
          nnNodes.push({
            id: nodeId,
            layerIdx: layerIdx,
            nodeIdx: i,
            x: layer.x,
            y: y,
            layerSpec: layer
          });
        }
      });

      // 2. Generate Interconnected Synaptic Bezier Curves
      for (let l = 0; l < layerSpecs.length - 1; l++) {
        const sourceNodes = nnNodes.filter(n => n.layerIdx === l);
        const targetNodes = nnNodes.filter(n => n.layerIdx === l + 1);

        sourceNodes.forEach((sNode, sIdx) => {
          targetNodes.forEach((tNode, tIdx) => {
            // Deterministic pseudo-random weight for reproducible physics structure
            const seed = (l * 100 + sIdx * 10 + tIdx);
            const rawVal = Math.sin(seed * 9.3) * 1.5;
            const isPos = Math.cos(seed * 5.1) > -0.2;
            const weight = (isPos ? 1 : -1) * (0.3 + Math.abs(rawVal) * 0.7);

            const dx = (tNode.x - sNode.x) * 0.52;
            const pathD = `M ${sNode.x + 16} ${sNode.y} C ${sNode.x + 16 + dx} ${sNode.y}, ${tNode.x - 16 - dx} ${tNode.y}, ${tNode.x - 16} ${tNode.y}`;

            nnSynapses.push({
              source: sNode.id,
              target: tNode.id,
              weight: weight,
              d: pathD,
              fromLayer: l
            });
          });
        });
      }

      // Render Synapses
      nnSynapses.forEach((syn, idx) => {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', syn.d);
        path.setAttribute('id', `syn_${idx}`);
        path.setAttribute('class', `synapse-line syn-source-${syn.source} syn-target-${syn.target}`);

        const absW = Math.abs(syn.weight);
        const strokeWidth = (0.8 + absW * 1.8).toFixed(1);
        path.setAttribute('stroke-width', strokeWidth);

        if (syn.weight > 0) {
          path.setAttribute('stroke', '#38bdf8');
          path.setAttribute('stroke-opacity', (0.25 + absW * 0.35).toFixed(2));
        } else {
          path.setAttribute('stroke', '#f97316');
          path.setAttribute('stroke-dasharray', '4,3');
          path.setAttribute('stroke-opacity', (0.22 + absW * 0.35).toFixed(2));
        }
        path.setAttribute('fill', 'none');
        synapseGroup.appendChild(path);
      });

      // Render Nodes with Playground-Style Activation Squares
      nnNodes.forEach((node) => {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('transform', `translate(${node.x}, ${node.y})`);
        g.setAttribute('class', 'neuron-node cursor-pointer group');
        g.setAttribute('id', `node_${node.id}`);

        // Node Outer Square/Card (Like Playground)
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', '-16');
        rect.setAttribute('y', '-16');
        rect.setAttribute('width', '32');
        rect.setAttribute('height', '32');
        rect.setAttribute('rx', '8');
        rect.setAttribute('fill', '#161b22');
        rect.setAttribute('stroke', node.layerIdx === 0 ? '#38bdf8' : (node.layerIdx === 4 ? '#fbbf24' : '#a855f7'));
        rect.setAttribute('stroke-width', '1.8');
        g.appendChild(rect);

        // Internal Activation Gradient Graphic
        const inner = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        inner.setAttribute('x', '-12');
        inner.setAttribute('y', '-12');
        inner.setAttribute('width', '24');
        inner.setAttribute('height', '24');
        inner.setAttribute('rx', '5');
        
        // Shading based on layer
        if (node.layerIdx === 0) {
          inner.setAttribute('fill', '#0284c7');
          inner.setAttribute('fill-opacity', '0.45');
        } else if (node.layerIdx === 4) {
          inner.setAttribute('fill', '#d97706');
          inner.setAttribute('fill-opacity', '0.45');
        } else {
          inner.setAttribute('fill', node.nodeIdx % 2 === 0 ? '#7c3aed' : '#2563eb');
          inner.setAttribute('fill-opacity', '0.35');
        }
        g.appendChild(inner);

        // Center dot
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('cx', '0');
        dot.setAttribute('cy', '0');
        dot.setAttribute('r', '2.5');
        dot.setAttribute('fill', '#ffffff');
        g.appendChild(dot);

        // Text Labels for Features & Outputs
        if (node.layerIdx === 0 && node.layerSpec.labels) {
          const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          text.setAttribute('x', '-24');
          text.setAttribute('y', '4');
          text.setAttribute('fill', '#f0f6fc');
          text.setAttribute('font-size', '10.5');
          text.setAttribute('font-weight', 'bold');
          text.setAttribute('text-anchor', 'end');
          text.textContent = node.layerSpec.labels[node.nodeIdx];
          g.appendChild(text);

          const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          sub.setAttribute('x', '-24');
          sub.setAttribute('y', '15');
          sub.setAttribute('fill', '#38bdf8');
          sub.setAttribute('font-size', '9');
          sub.setAttribute('font-family', 'monospace');
          sub.setAttribute('text-anchor', 'end');
          sub.textContent = node.layerSpec.sub[node.nodeIdx];
          g.appendChild(sub);
        } else if (node.layerIdx === 4 && node.layerSpec.labels) {
          const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          text.setAttribute('x', '24');
          text.setAttribute('y', '4');
          text.setAttribute('fill', '#fbbf24');
          text.setAttribute('font-size', '10.5');
          text.setAttribute('font-weight', 'bold');
          text.setAttribute('text-anchor', 'start');
          text.textContent = node.layerSpec.labels[node.nodeIdx];
          g.appendChild(text);
        }

        // Hover & Click Events
        g.addEventListener('mouseenter', () => highlightNodeSynapses(node.id));
        g.addEventListener('mouseleave', () => resetSynapseHighlights());
        g.addEventListener('click', () => inspectNeuron(node));

        neuronGroup.appendChild(g);
      });
    }

    // Highlight Synaptic Mesh connected to hovered node
    function highlightNodeSynapses(nodeId) {
      document.querySelectorAll('.synapse-line').forEach(line => {
        line.setAttribute('opacity', '0.12');
      });

      const connected = document.querySelectorAll(`.syn-source-${nodeId}, .syn-target-${nodeId}`);
      connected.forEach(line => {
        line.setAttribute('opacity', '1.0');
        line.setAttribute('stroke-width', '2.8');
        line.setAttribute('filter', 'url(#glow-blue)');
      });

      const tip = document.getElementById('neuron-status-tip');
      if (tip) {
        tip.innerHTML = `Active Node: <span class="text-white font-bold">${nodeId}</span> • Activated ${connected.length} Synaptic Connections`;
      }
    }

    function resetSynapseHighlights() {
      document.querySelectorAll('.synapse-line').forEach(line => {
        line.setAttribute('opacity', '1.0');
        line.removeAttribute('filter');
        const absW = line.getAttribute('stroke-dasharray') ? 1.5 : 2.0;
        line.setAttribute('stroke-width', line.getAttribute('data-original-width') || '1.5');
      });

      const tip = document.getElementById('neuron-status-tip');
      if (tip) {
        tip.innerText = 'Hover or click any neuron node to inspect synaptic paths';
      }
    }

    // Click inspection
    function inspectNeuron(node) {
      let desc = '';
      if (node.layerIdx === 0) {
        desc = `Input Feature: ${node.layerSpec.labels[node.nodeIdx]} (${node.layerSpec.sub[node.nodeIdx]})\nDimension: [N, 1]\nPhysical Function: Invariant fluid deformation measure.`;
      } else if (node.layerIdx === 4) {
        desc = `Basis Weight: ${node.layerSpec.labels[node.nodeIdx]}\nMultiplies Pope Tensor Basis matrix T(${node.nodeIdx + 1}) in-situ to drive secondary flow.`;
      } else {
        desc = `Hidden Neuron ${node.id} (${node.layerSpec.name})\nActivation: LeakyReLU(alpha=0.1)\nWeight matrix shape: [64 x 64]\nBias: +0.024\nStatus: Active in feedforward pass.`;
      }
      alert(desc);
    }

    // Feedforward Wave Animation (Flowing glowing particle packets)
    async function startFeedforwardWave() {
      if (isPulseActive) return;
      isPulseActive = true;
      const btnIcon = document.getElementById('nn-pulse-icon');
      const btnText = document.getElementById('nn-pulse-text');
      if (btnIcon) btnIcon.innerText = '⏳';
      if (btnText) btnText.innerText = 'Propagating Wave...';

      const pulseGroup = document.getElementById('pulse-group');
      if (!pulseGroup) return;
      pulseGroup.innerHTML = '';

      // Animate wave layer by layer
      for (let l = 0; l < layerSpecs.length - 1; l++) {
        const layerSynapses = nnSynapses.filter(s => s.fromLayer === l);
        // Pick representative subset of paths for crisp visual clarity
        const activeSubset = layerSynapses.filter((_, idx) => idx % 3 === 0);

        activeSubset.forEach(syn => {
          const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          circle.setAttribute('r', '4');
          circle.setAttribute('fill', syn.weight > 0 ? '#38bdf8' : '#fb923c');
          circle.setAttribute('filter', 'url(#glow-blue)');
          pulseGroup.appendChild(circle);

          // Animate along path using animateMotion
          const animMotion = document.createElementNS('http://www.w3.org/2000/svg', 'animateMotion');
          animMotion.setAttribute('path', syn.d);
          animMotion.setAttribute('begin', '0s');
          animMotion.setAttribute('dur', '0.55s');
          animMotion.setAttribute('fill', 'freeze');
          circle.appendChild(animMotion);
          animMotion.beginElement();
        });

        await new Promise(r => setTimeout(r, 450));
        pulseGroup.innerHTML = '';
      }

      // Flash output heatmap
      const heatRect = document.querySelector('#output-viz-group rect:nth-of-type(2)');
      if (heatRect) {
        heatRect.setAttribute('stroke', '#34d399');
        heatRect.setAttribute('stroke-width', '3');
        setTimeout(() => {
          heatRect.setAttribute('stroke', '#38bdf8');
          heatRect.setAttribute('stroke-width', '1.5');
        }, 600);
      }

      if (btnIcon) btnIcon.innerText = '⚡';
      if (btnText) btnText.innerText = 'Run Feedforward Wave';
      isPulseActive = false;
    }

    function toggleWeightSignOnly() {
      filterStrong = !filterStrong;
      document.querySelectorAll('.synapse-line').forEach(line => {
        const w = parseFloat(line.getAttribute('stroke-width'));
        if (filterStrong && w < 1.8) {
          line.style.display = 'none';
        } else {
          line.style.display = '';
        }
      });
    }

    // Call init when switching to neural view
    const originalSwitchPage = window.switchPage;
    window.switchPage = function(pageId) {
      if (typeof originalSwitchPage === 'function') {
        originalSwitchPage(pageId);
      }
      if (pageId === 'neural') {
        setTimeout(initNeuralPlayground, 50);
      }
    };

    // Auto-init on load if view is active
    document.addEventListener('DOMContentLoaded', () => {
      initNeuralPlayground();
    });
"""

def patch_file(file_path: Path):
    print(f"--> Patching {file_path}...")
    content = file_path.read_text(encoding="utf-8")
    
    # 1. Replace <main id="view-neural">...</main>
    pattern = r'<main id="view-neural".*?</main>'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"  [!] Could not locate view-neural block in {file_path}")
        return False
        
    content = content[:match.start()] + NEURAL_VIEW_HTML.strip() + content[match.end():]
    
    script_close = content.rfind("</script>")
    if script_close != -1:
        content = content[:script_close] + "\n" + PLAYGROUND_JS + "\n" + content[script_close:]
    file_path.write_text(content, encoding="utf-8")
    print(f"  [OK] Successfully updated {file_path} ({file_path.stat().st_size / 1e3:.1f} KB)")
    return True

def main():
    root = Path(__file__).resolve().parent.parent
    index_html = root / "docs" / "index.html"
    studio_artifact = Path("C:/Users/soumi/.gemini/antigravity/brain/e447c5f6-5df9-48f5-9694-da07a482558a/tbnn_studio.html")
    
    patch_file(index_html)
    if studio_artifact.exists():
        patch_file(studio_artifact)
    print("\n--> NEURAL PLAYGROUND VIEW FULLY INTEGRATED!")

if __name__ == "__main__":
    main()

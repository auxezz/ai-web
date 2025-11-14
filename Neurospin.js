// Neurospin.js - central Neuro animation + typing sound helper
(function(window){
  // Minimal, self-contained NeuroSpin module
  const NeuroSpin = {
    _neuroEl: null,
    _audioCtx: null,
    _typingInterval: null,

    init(rootId = 'neuro-corner') {
      this._neuroEl = document.getElementById(rootId);
      if (!this._neuroEl) {
        console.warn(`NeuroSpin: element with id '${rootId}' not found.`);
      }
      // lazy audio context (created on first sound to avoid autoplay issues)
      this._audioCtx = null;
      // start spinning by default so Neuro slowly rotates
      try {
        this.startSpinning();
      } catch (e) {
        console.warn('NeuroSpin.startSpinning failed', e);
      }
    },

    startSpinning() {
      if (this._neuroEl) {
        this._neuroEl.classList.add('spinning');
        console.debug('NeuroSpin: started spinning on', this._neuroEl);
      } else {
        console.debug('NeuroSpin: startSpinning called but no element');
      }
    },

    stopSpinning() {
      if (this._neuroEl) this._neuroEl.classList.remove('spinning');
    },

    _ensureAudio() {
      if (!this._audioCtx) {
        this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
    },

    playBeep(volume = 0.08, freq = 800, duration = 0.05) {
      try {
        this._ensureAudio();
        const osc = this._audioCtx.createOscillator();
        const gain = this._audioCtx.createGain();
        osc.type = 'square';
        osc.frequency.value = freq;
        osc.connect(gain);
        gain.connect(this._audioCtx.destination);
        gain.gain.setValueAtTime(volume, this._audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this._audioCtx.currentTime + duration);
        osc.start(this._audioCtx.currentTime);
        osc.stop(this._audioCtx.currentTime + duration);
      } catch (err) {
        // audio errors should not break the app
      }
    },

    startTalking(beepIntervalMs = 100) {
      if (!this._neuroEl) return;
      // If Neuro was spinning, remember that and stop spinning while talking
      this._wasSpinning = this._neuroEl.classList.contains('spinning');
      if (this._wasSpinning) this._neuroEl.classList.remove('spinning');
      this._neuroEl.classList.add('talking');
      // start interval for beeps
      if (this._typingInterval) return; // already running
      this._typingInterval = setInterval(() => this.playBeep(), beepIntervalMs);
    },

    stopTalking() {
      if (!this._neuroEl) return;
      this._neuroEl.classList.remove('talking');
      // restore spinning state if it was spinning before talking
      if (this._wasSpinning) {
        this._neuroEl.classList.add('spinning');
        this._wasSpinning = false;
      }
      if (this._typingInterval) {
        clearInterval(this._typingInterval);
        this._typingInterval = null;
      }
    },

    // Animate text character-by-character while Neuro is talking
    // element: DOM element to write into
    // prefix: prefix text (e.g. 'Neuro: ')
    // text: the full response text
    // charDelay: ms per character (default 50)
    animateText(element, prefix, text, charDelay = 50) {
      return new Promise((resolve) => {
        if (!element) return resolve();
        const totalDuration = Math.max(200, text.length * charDelay); // safety min

        let index = 0;
        element.textContent = prefix;

        // start animation + audio
        this.startTalking(Math.max(75, Math.floor(charDelay * 1.5)));

        const interval = setInterval(() => {
          if (index < text.length) {
            element.textContent = prefix + text.substring(0, index + 1);
            // play occasional beep for character groups
            if (index % 2 === 0) this.playBeep();
            index++;
            element.parentElement && (element.parentElement.scrollTop = element.parentElement.scrollHeight);
          } else {
            clearInterval(interval);
            // small delay to let the last beep / bob finish
            setTimeout(() => {
              this.stopTalking();
              resolve();
            }, 80);
          }
        }, charDelay);
      });
    }
  };

  // expose globally
  window.NeuroSpin = NeuroSpin;
})(window);

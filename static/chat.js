// static/chat.js (updated: render HTML & embed YouTube Shorts)
document.addEventListener("DOMContentLoaded", () => {
  const messagesEl = document.getElementById("messages");
  const optionsEl = document.getElementById("options");
  const inputForm = document.getElementById("input-form");
  const inputEl = document.getElementById("message-input");
  const startupGreetingEl = document.getElementById("startup-greeting");

  function appendMessage(text, who = "bot", allowHTML = false) {
    const wrap = document.createElement("div");
    wrap.className = who + " message";
    const b = document.createElement("div");
    b.className = "bubble";
    if (allowHTML) {
      b.innerHTML = text;
      // attach handlers for yt-short anchors inside this bubble
      const anchors = b.querySelectorAll("a.yt-short");
      anchors.forEach(a => {
        a.addEventListener("click", (ev) => {
          ev.preventDefault();
          const href = a.getAttribute("href");
          showEmbeddedShort(href, a.textContent || "Video");
        });
      });
    } else {
      b.textContent = text;
    }
    wrap.appendChild(b);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function showEmbeddedShort(shortUrl, title = "") {
    let vid = null;
    try {
      const url = new URL(shortUrl);
      const pathParts = url.pathname.split("/");
      const idx = pathParts.indexOf("shorts");
      if (idx >= 0 && pathParts.length > idx + 1) {
        vid = pathParts[idx + 1];
      } else {
        vid = url.searchParams.get("v");
      }
    } catch (e) {
      const m = shortUrl.match(/shorts\/([A-Za-z0-9_-]+)/);
      if (m) vid = m[1];
    }
    if (!vid) {
      appendMessage("Sorry — couldn't open that video. Invalid link.", "bot");
      return;
    }

    const embed = `https://www.youtube.com/embed/${vid}?autoplay=1&controls=1&rel=0`;
    const wrap = document.createElement("div");
    wrap.className = "bot message";
    const b = document.createElement("div");
    b.className = "bubble video-bubble";
    b.innerHTML = `<div class="video-title">${escapeHtml(title)}</div>
                   <div class="video-wrapper">
                     <iframe src="${embed}" title="YouTube Short" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
                   </div>`;
    wrap.appendChild(b);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escapeHtml(unsafe) {
    return String(unsafe).replace(/[&<"'>]/g, function(m) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' })[m];
    });
  }

  function showOptions(options) {
    optionsEl.innerHTML = "";
    if (!options || options.length === 0) return;
    options.forEach(opt => {
      const btn = document.createElement("button");
      btn.className = "option-btn";
      btn.textContent = opt.label;
      btn.type = "button";
      btn.addEventListener("click", () => selectOption(opt));
      optionsEl.appendChild(btn);
    });
  }

  async function postMessage(payload) {
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        throw new Error("Network error: " + res.status);
      }
      return await res.json();
    } catch (err) {
      console.error("chat post failed", err);
      return { bot_message: "Sorry — I couldn't reach the server. Try again.", options: [] };
    }
  }

  async function selectOption(opt) {
    appendMessage(opt.label, "user");
    const resp = await postMessage({ option_tag: opt.tag });
    const isHtml = /<a\s+class="yt-short"/i.test(resp.bot_message) || /<div|<ol|<li/i.test(resp.bot_message);
    appendMessage(resp.bot_message, "bot", isHtml);
    showOptions(resp.options || []);
  }

  async function sendMessage(event) {
    if (event && event.preventDefault) event.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    appendMessage(text, "user");
    inputEl.value = "";
    inputEl.disabled = true;

    const resp = await postMessage({ message: text });
    const isHtml = /<a\s+class="yt-short"/i.test(resp.bot_message) || /<div|<ol|<li/i.test(resp.bot_message);
    appendMessage(resp.bot_message, "bot", isHtml);
    showOptions(resp.options || []);
    inputEl.disabled = false;
    inputEl.focus();
  }

  inputForm.addEventListener("submit", sendMessage);

  (async function init() {
    if (startupGreetingEl && startupGreetingEl.textContent.trim()) {
      appendMessage(startupGreetingEl.textContent.trim() + "!", "bot");
    }
    const resp = await postMessage({ message: "" });
    const isHtml = /<a\s+class="yt-short"/i.test(resp.bot_message) || /<div|<ol|<li/i.test(resp.bot_message);
    appendMessage(resp.bot_message, "bot", isHtml);
    showOptions(resp.options || []);
  })();
});


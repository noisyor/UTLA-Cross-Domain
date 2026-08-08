const header = document.querySelector("[data-header]");
const menuButton = document.querySelector(".menu-toggle");
const nav = document.querySelector(".site-nav");

const updateHeader = () => header?.classList.toggle("scrolled", window.scrollY > 20);
updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

menuButton?.addEventListener("click", () => {
  const open = menuButton.getAttribute("aria-expanded") === "true";
  menuButton.setAttribute("aria-expanded", String(!open));
  nav?.classList.toggle("open", !open);
  document.body.style.overflow = open ? "" : "hidden";
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    menuButton?.setAttribute("aria-expanded", "false");
    nav.classList.remove("open");
    document.body.style.overflow = "";
  });
});

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    const filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");

    document.querySelectorAll("[data-category]").forEach((card) => {
      const categories = card.dataset.category.split(" ");
      card.hidden = filter !== "all" && !categories.includes(filter);
    });
  });
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select text";
    }
    window.setTimeout(() => { button.textContent = original; }, 1600);
  });
});

const traceCanvases = document.querySelectorAll("[data-trace]");

const drawSideChannelTrace = (canvas) => {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * pixelRatio);
  canvas.height = Math.round(rect.height * pixelRatio);

  const context = canvas.getContext("2d");
  context.scale(pixelRatio, pixelRatio);
  context.clearRect(0, 0, rect.width, rect.height);

  const target = canvas.dataset.trace === "target";
  const stroke = getComputedStyle(canvas.parentElement).color;
  const gaussian = (x, center, spread) => Math.exp(-0.5 * ((x - center) / spread) ** 2);
  const sample = (time, offset = 0) => {
    const shifted = time + (target ? 0.035 : 0) + offset * 0.0015;
    const active = gaussian(shifted, 0.5, 0.25);
    const carrier = Math.sin(shifted * Math.PI * (target ? 54 : 49)) * 0.085 * active;
    const fineLeakage = Math.sin(shifted * Math.PI * (target ? 121 : 113) + offset) * 0.026 * active;
    const operations =
      -0.22 * gaussian(shifted, 0.22, 0.017) +
      0.17 * gaussian(shifted, 0.285, 0.022) -
      0.26 * gaussian(shifted, 0.47, 0.014) +
      0.2 * gaussian(shifted, 0.535, 0.019) -
      0.18 * gaussian(shifted, 0.72, 0.025);
    const drift = 0.025 * Math.sin(shifted * Math.PI * 5 + (target ? 0.9 : 0));
    return 0.5 + carrier + fineLeakage + operations + drift;
  };

  const plot = (offset, alpha, lineWidth) => {
    context.beginPath();
    for (let index = 0; index <= 220; index += 1) {
      const time = index / 220;
      const x = time * rect.width;
      const y = sample(time, offset) * rect.height;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.globalAlpha = alpha;
    context.strokeStyle = stroke;
    context.lineWidth = lineWidth;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.stroke();
  };

  plot(-2, 0.16, 1);
  plot(2, 0.18, 1);
  plot(0, 0.95, 1.8);
  context.globalAlpha = 1;
};

traceCanvases.forEach(drawSideChannelTrace);

if (traceCanvases.length && "ResizeObserver" in window) {
  const traceObserver = new ResizeObserver((entries) => {
    entries.forEach((entry) => drawSideChannelTrace(entry.target));
  });
  traceCanvases.forEach((canvas) => traceObserver.observe(canvas));
} else if (traceCanvases.length) {
  window.addEventListener("resize", () => traceCanvases.forEach(drawSideChannelTrace));
}

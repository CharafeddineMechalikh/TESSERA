// The fences render as <pre class="tessera-mermaid"><code>source</code></pre>.
// Mermaid's own start-up grabs every .mermaid element on load, reads innerHTML,
// and feeds its parser the wrapper tags, which leaves the block empty. Keeping a
// private class and calling render() with textContent avoids both problems.
function renderTesseraDiagrams() {
  if (!window.mermaid) {
    return;
  }
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: "default",
  });
  const nodes = Array.prototype.slice.call(
    document.querySelectorAll(".tessera-mermaid")
  );
  nodes.forEach(function (node, index) {
    if (node.getAttribute("data-tessera-state")) {
      return;
    }
    node.setAttribute("data-tessera-state", "rendered");
    const source = node.textContent.trim();
    const id = "tessera-diagram-" + index + "-" + Date.now();
    window.mermaid
      .render(id, source)
      .then(function (result) {
        const figure = document.createElement("div");
        figure.className = "tessera-mermaid-figure";
        figure.innerHTML = result.svg;
        node.replaceWith(figure);
      })
      .catch(function (error) {
        node.setAttribute("data-tessera-state", "failed");
        node.classList.add("tessera-mermaid-error");
        node.textContent =
          "diagram failed to render: " +
          String(error && error.message ? error.message : error);
      });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderTesseraDiagrams);
} else {
  renderTesseraDiagrams();
}

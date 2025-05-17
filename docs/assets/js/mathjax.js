window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document.addEventListener("DOMContentLoaded", function() {
  // Load MathJax script
  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
  script.async = true;

  // Ensure MathJax is reprocessed if dynamic content is loaded
  script.onload = function() {
    if (typeof MathJax !== 'undefined') {
      // Force MathJax to process the page on load
      setTimeout(function() {
        MathJax.typeset();
      }, 500);
    }
  };

  document.head.appendChild(script);
});

document.addEventListener('DOMContentLoaded', function() {
  // Add class to body when on index page
  const isIndexPage = document.querySelector('.md-content__inner > h1')?.textContent.trim() === 'Introduction';
  if (isIndexPage) {
    document.body.classList.add('index-page');
  }

  // Ensure navigation starts collapsed but active section is expanded
  function setupNavigationCollapsing() {
    // First collapse all sections
    document.querySelectorAll('.md-nav__item--nested input[type="checkbox"]').forEach(function(checkbox) {
      checkbox.checked = false;
    });

    // Then expand active section and its parents
    const activeItems = document.querySelectorAll('.md-nav__item--active');
    activeItems.forEach(function(activeItem) {
      let parent = activeItem.closest('.md-nav__item--nested');
      while (parent) {
        const checkbox = parent.querySelector('input[type="checkbox"]');
        if (checkbox) {
          checkbox.checked = true;
        }
        parent = parent.parentElement.closest('.md-nav__item--nested');
      }
    });
  }

  // Apply collapsing logic after a small delay to ensure all elements are loaded
  setTimeout(function() {
    setupNavigationCollapsing();
  }, 100);

  // Re-apply on hash changes or navigation events
  window.addEventListener('hashchange', function() {
    setupNavigationCollapsing();
  });

  // Handle Material instant navigation
  const content = document.querySelector('.md-content');
  if (content) {
    const observer = new MutationObserver(function() {
      setTimeout(function() {
        setupNavigationCollapsing();
      }, 100);
    });

    observer.observe(content, {
      childList: true,
      subtree: true
    });
  }
});

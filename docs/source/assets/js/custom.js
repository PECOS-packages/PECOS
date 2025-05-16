document.addEventListener('DOMContentLoaded', function() {
  // Add class to body when on index page
  const isIndexPage = document.querySelector('.md-content__inner > h1')?.textContent.trim() === 'Introduction';
  if (isIndexPage) {
    document.body.classList.add('index-page');
  }
});
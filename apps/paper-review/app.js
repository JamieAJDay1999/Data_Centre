(() => {
  'use strict';

  const paper = window.PAPER_DATA;
  if (!paper || !Array.isArray(paper.pages)) {
    document.body.innerHTML = '<p style="padding:2rem;font-family:sans-serif">Paper data could not be loaded. Re-run scripts/build_assets.py.</p>';
    return;
  }

  const STORAGE_KEY = 'data-centre-paper-review-comments-v1';
  const migration = window.PAPER_COMMENT_MIGRATION;
  const ZOOM_LEVELS = [0.72, 0.85, 1, 1.15, 1.35, 1.55];
  const BASE_PAGE_WIDTH = 820;
  const loadedComments = loadComments();

  const state = {
    comments: loadedComments.comments,
    migrationSummary: loadedComments.summary,
    filter: 'open',
    commentMode: false,
    selectedId: null,
    draft: null,
    zoomIndex: 2,
    currentPage: 1,
    toastTimer: null,
  };

  const els = {
    pages: document.querySelector('#pages'),
    documentTitle: document.querySelector('#documentTitle'),
    pageSelect: document.querySelector('#pageSelect'),
    pageTotal: document.querySelector('#pageTotal'),
    zoomOutBtn: document.querySelector('#zoomOutBtn'),
    zoomInBtn: document.querySelector('#zoomInBtn'),
    zoomLabel: document.querySelector('#zoomLabel'),
    addCommentBtn: document.querySelector('#addCommentBtn'),
    toggleNotesBtn: document.querySelector('#toggleNotesBtn'),
    closeSidebarBtn: document.querySelector('#closeSidebarBtn'),
    modeBanner: document.querySelector('#modeBanner'),
    modeMessage: document.querySelector('#modeMessage'),
    saveStatus: document.querySelector('#saveStatus'),
    headerCount: document.querySelector('#headerCount'),
    openCount: document.querySelector('#openCount'),
    resolvedCount: document.querySelector('#resolvedCount'),
    commentsList: document.querySelector('#commentsList'),
    editorPanel: document.querySelector('#editorPanel'),
    listPanel: document.querySelector('#listPanel'),
    exportBtn: document.querySelector('#exportBtn'),
    importBtn: document.querySelector('#importBtn'),
    importInput: document.querySelector('#importInput'),
    toast: document.querySelector('#toast'),
  };

  initialise();

  function initialise() {
    document.title = `Paper Review - ${paper.shortTitle || paper.title}`;
    els.documentTitle.textContent = paper.shortTitle || paper.title;
    els.pageTotal.textContent = `of ${paper.pageCount}`;
    buildPageSelector();
    renderPages();
    renderAllComments();
    bindEvents();
    observePages();
    updateZoom();
    syncSidebarMode();
    if (state.migrationSummary.checked) {
      const { kept, removed } = state.migrationSummary;
      window.setTimeout(() => showToast(`Paper updated: ${kept} comment${kept === 1 ? '' : 's'} kept, ${removed} removed`), 0);
    }
  }

  function buildPageSelector() {
    const fragment = document.createDocumentFragment();
    for (let pageNumber = 1; pageNumber <= paper.pageCount; pageNumber += 1) {
      const option = document.createElement('option');
      option.value = pageNumber;
      option.textContent = pageNumber;
      fragment.append(option);
    }
    els.pageSelect.replaceChildren(fragment);
  }

  function renderPages() {
    const fragment = document.createDocumentFragment();
    paper.pages.forEach((page) => {
      const pageElement = document.createElement('section');
      pageElement.className = 'paper-page';
      pageElement.dataset.page = page.number;
      pageElement.dataset.pageLabel = `PAGE ${page.number}`;
      pageElement.id = `page-${page.number}`;
      pageElement.setAttribute('aria-label', `Page ${page.number} of ${paper.pageCount}`);
      pageElement.style.aspectRatio = `${page.width} / ${page.height}`;

      const image = document.createElement('img');
      image.className = 'page-image';
      image.src = page.image;
      image.alt = '';
      image.draggable = false;
      image.loading = page.number <= 2 ? 'eager' : 'lazy';
      image.decoding = 'async';

      const textLayer = document.createElement('div');
      textLayer.className = 'text-layer';
      textLayer.setAttribute('aria-label', `Selectable text for page ${page.number}`);
      page.words.forEach((word) => {
        const token = document.createElement('span');
        token.className = 'text-token';
        token.textContent = `${word.text} `;
        token.style.left = `${word.x}%`;
        token.style.top = `${word.y}%`;
        token.style.width = `${Math.max(word.w + 0.25, 0.35)}%`;
        token.style.height = `${Math.max(word.h + 0.15, 0.35)}%`;
        token.style.fontSize = `${Math.max((word.h / page.width) * 95, 0.45)}cqw`;
        textLayer.append(token);
      });

      const annotationLayer = document.createElement('div');
      annotationLayer.className = 'annotation-layer';
      annotationLayer.dataset.page = page.number;

      pageElement.append(image, textLayer, annotationLayer);
      fragment.append(pageElement);
    });
    els.pages.replaceChildren(fragment);
  }

  function bindEvents() {
    els.addCommentBtn.addEventListener('click', () => setCommentMode(!state.commentMode));
    els.pageSelect.addEventListener('change', () => scrollToPage(Number(els.pageSelect.value)));
    els.zoomOutBtn.addEventListener('click', () => changeZoom(-1));
    els.zoomInBtn.addEventListener('click', () => changeZoom(1));
    els.toggleNotesBtn.addEventListener('click', toggleSidebar);
    els.closeSidebarBtn.addEventListener('click', closeSidebar);
    els.exportBtn.addEventListener('click', exportComments);
    els.importBtn.addEventListener('click', () => els.importInput.click());
    els.importInput.addEventListener('change', importComments);
    window.addEventListener('resize', syncSidebarMode);

    els.pages.addEventListener('click', (event) => {
      const marker = event.target.closest('.comment-marker');
      if (marker) {
        event.stopPropagation();
        openComment(marker.dataset.id, true);
        return;
      }
      if (!state.commentMode) return;
      const pageElement = event.target.closest('.paper-page');
      if (!pageElement) return;
      const rect = pageElement.getBoundingClientRect();
      const x = clamp(((event.clientX - rect.left) / rect.width) * 100, 0, 100);
      const y = clamp(((event.clientY - rect.top) / rect.height) * 100, 0, 100);
      startDraft(Number(pageElement.dataset.page), x, y);
    });

    document.querySelectorAll('.filter-chip').forEach((button) => {
      button.addEventListener('click', () => {
        state.filter = button.dataset.filter;
        document.querySelectorAll('.filter-chip').forEach((chip) => chip.classList.toggle('is-active', chip === button));
        renderCommentList();
      });
    });

    document.addEventListener('keydown', (event) => {
      const editing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
      if (event.key === 'Escape') {
        if (state.draft) cancelEditor();
        else if (state.commentMode) setCommentMode(false);
      }
      if (!editing && event.key.toLowerCase() === 'c') {
        event.preventDefault();
        setCommentMode(!state.commentMode);
      }
      if (!editing && event.key === '-') changeZoom(-1);
      if (!editing && (event.key === '+' || event.key === '=')) changeZoom(1);
    });
  }

  function observePages() {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (!visible.length) return;
      state.currentPage = Number(visible[0].target.dataset.page);
      els.pageSelect.value = String(state.currentPage);
    }, { rootMargin: '-25% 0px -55% 0px', threshold: [0, 0.15, 0.35, 0.6] });

    document.querySelectorAll('.paper-page').forEach((page) => observer.observe(page));
  }

  function setCommentMode(enabled) {
    state.commentMode = enabled;
    document.body.classList.toggle('is-commenting', enabled);
    els.addCommentBtn.classList.toggle('is-active', enabled);
    els.addCommentBtn.setAttribute('aria-pressed', String(enabled));
    els.modeBanner.classList.toggle('is-commenting', enabled);
    els.modeMessage.innerHTML = enabled
      ? '<strong>Pin mode is on.</strong> Click the exact place on a page where your comment belongs. Press Esc to cancel.'
      : 'Select <strong>Add comment</strong>, then click anywhere on a page to pin a note.';
  }

  function startDraft(page, x, y) {
    if (state.draft) cancelEditor();
    state.draft = {
      id: createId(),
      page,
      x: round(x),
      y: round(y),
      body: '',
      resolved: false,
      anchor: createTextAnchor(paper.pages, page, x, y),
      documentFingerprint: paper.fingerprint,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      isDraft: true,
    };
    state.selectedId = state.draft.id;
    setCommentMode(false);
    openSidebar();
    renderMarkers();
    renderEditor(state.draft, true);
  }

  function renderAllComments() {
    renderMarkers();
    renderCommentList();
    updateCounts();
  }

  function renderMarkers() {
    document.querySelectorAll('.annotation-layer').forEach((layer) => layer.replaceChildren());
    const comments = orderedComments();
    const numberById = new Map(comments.map((comment, index) => [comment.id, index + 1]));
    const all = state.draft ? [...comments, state.draft] : comments;

    all.forEach((comment) => {
      const layer = document.querySelector(`.annotation-layer[data-page="${comment.page}"]`);
      if (!layer) return;
      const marker = document.createElement('button');
      marker.type = 'button';
      marker.className = 'comment-marker';
      marker.classList.toggle('is-resolved', Boolean(comment.resolved));
      marker.classList.toggle('is-selected', comment.id === state.selectedId);
      marker.classList.toggle('is-draft', Boolean(comment.isDraft));
      marker.dataset.id = comment.id;
      marker.style.left = `${comment.x}%`;
      marker.style.top = `${comment.y}%`;
      marker.textContent = comment.isDraft ? '+' : numberById.get(comment.id);
      marker.setAttribute('aria-label', comment.isDraft ? `New comment on page ${comment.page}` : `Comment ${numberById.get(comment.id)} on page ${comment.page}: ${comment.body}`);
      marker.title = comment.body || 'New comment';
      layer.append(marker);
    });
  }

  function renderCommentList() {
    const filtered = orderedComments().filter((comment) => {
      if (state.filter === 'open') return !comment.resolved;
      if (state.filter === 'resolved') return comment.resolved;
      return true;
    });

    if (!filtered.length) {
      const messages = {
        open: ['No open comments', 'Pin a comment to the paper or switch to All to review resolved notes.'],
        resolved: ['No resolved comments', 'Mark a finished note as resolved and it will appear here.'],
        all: ['No comments yet', 'Select Add comment and click the exact location you want to discuss.'],
      };
      const [title, body] = messages[state.filter];
      els.commentsList.innerHTML = `<div class="empty-state"><div class="empty-state-icon">+</div><strong>${title}</strong><p>${body}</p></div>`;
      return;
    }

    const allOrdered = orderedComments();
    els.commentsList.replaceChildren(...filtered.map((comment) => {
      const index = allOrdered.findIndex((item) => item.id === comment.id) + 1;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'comment-card';
      button.classList.toggle('is-resolved', Boolean(comment.resolved));
      button.classList.toggle('is-selected', comment.id === state.selectedId);
      const meta = document.createElement('div');
      meta.className = 'comment-meta';
      const number = document.createElement('span');
      number.className = 'comment-number';
      number.textContent = `#${index} · Page ${comment.page}`;
      const date = document.createElement('span');
      date.textContent = formatDate(comment.updatedAt);
      meta.append(number, date);
      const body = document.createElement('p');
      body.textContent = comment.body;
      button.append(meta, body);
      button.addEventListener('click', () => openComment(comment.id, true));
      return button;
    }));
  }

  function renderEditor(comment, isNew = false) {
    els.listPanel.hidden = true;
    els.editorPanel.hidden = false;
    els.editorPanel.innerHTML = `
      <button class="editor-back" id="editorBackBtn" type="button">← Back to comments</button>
      <div class="editor-location">Page ${comment.page} · Pinned location</div>
      <h3>${isNew ? 'Add a review note' : 'Edit comment'}</h3>
      <form id="commentForm">
        <label for="commentBody">Comment</label>
        <textarea id="commentBody" required maxlength="4000" placeholder="What should change here?">${escapeHtml(comment.body)}</textarea>
        <p class="editor-hint">This note is anchored to the point you selected and is saved only in this browser until exported.</p>
        ${isNew ? '' : `<label class="status-row"><input id="resolvedCheckbox" type="checkbox" ${comment.resolved ? 'checked' : ''}> Mark as resolved</label>`}
        <div class="editor-actions">
          <button class="button button-primary" type="submit">${isNew ? 'Add comment' : 'Save changes'}</button>
          <button class="button button-quiet" id="cancelEditorBtn" type="button">Cancel</button>
        </div>
        ${isNew ? '' : '<button class="danger-button" id="deleteCommentBtn" type="button">Delete comment</button>'}
      </form>`;

    const bodyInput = els.editorPanel.querySelector('#commentBody');
    requestAnimationFrame(() => bodyInput.focus());
    els.editorPanel.querySelector('#editorBackBtn').addEventListener('click', cancelEditor);
    els.editorPanel.querySelector('#cancelEditorBtn').addEventListener('click', cancelEditor);
    els.editorPanel.querySelector('#commentForm').addEventListener('submit', (event) => {
      event.preventDefault();
      const body = bodyInput.value.trim();
      if (!body) return;
      if (isNew) {
        const saved = { ...state.draft, body, isDraft: undefined, updatedAt: new Date().toISOString() };
        delete saved.isDraft;
        state.comments.push(saved);
        state.draft = null;
        state.selectedId = saved.id;
      } else {
        const target = state.comments.find((item) => item.id === comment.id);
        if (!target) return;
        target.body = body;
        target.resolved = Boolean(els.editorPanel.querySelector('#resolvedCheckbox')?.checked);
        target.updatedAt = new Date().toISOString();
      }
      persistComments();
      showList();
      renderAllComments();
      showToast(isNew ? 'Comment added' : 'Comment updated');
    });

    els.editorPanel.querySelector('#deleteCommentBtn')?.addEventListener('click', () => {
      if (!window.confirm('Delete this comment? This cannot be undone unless you exported a backup.')) return;
      state.comments = state.comments.filter((item) => item.id !== comment.id);
      state.selectedId = null;
      persistComments();
      showList();
      renderAllComments();
      showToast('Comment deleted');
    });
  }

  function openComment(id, scroll = false) {
    const comment = state.comments.find((item) => item.id === id);
    if (!comment) return;
    state.selectedId = id;
    openSidebar();
    renderMarkers();
    renderCommentList();
    renderEditor(comment, false);
    if (scroll) scrollToMarker(comment);
  }

  function cancelEditor() {
    state.draft = null;
    state.selectedId = null;
    showList();
    renderMarkers();
    renderCommentList();
  }

  function showList() {
    els.editorPanel.hidden = true;
    els.editorPanel.replaceChildren();
    els.listPanel.hidden = false;
  }

  function updateCounts() {
    const open = state.comments.filter((comment) => !comment.resolved).length;
    const resolved = state.comments.length - open;
    els.openCount.textContent = open;
    els.resolvedCount.textContent = resolved;
    els.headerCount.textContent = state.comments.length;
  }

  function orderedComments() {
    return [...state.comments].sort((a, b) => {
      if (a.page !== b.page) return a.page - b.page;
      if (a.y !== b.y) return a.y - b.y;
      return a.x - b.x;
    });
  }

  function persistComments() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.comments));
    els.saveStatus.textContent = 'Saved just now';
    window.setTimeout(() => { els.saveStatus.textContent = 'Saved in this browser'; }, 1800);
  }

  function loadComments() {
    try {
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      if (!Array.isArray(stored)) return { comments: [], summary: emptyMigrationSummary() };
      const prepared = prepareComments(stored);
      if (prepared.summary.checked) {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(prepared.comments)); } catch { /* Keep the migrated in-memory copy. */ }
      }
      return prepared;
    } catch {
      return { comments: [], summary: emptyMigrationSummary() };
    }
  }

  function emptyMigrationSummary() {
    return { checked: 0, kept: 0, removed: 0 };
  }

  function prepareComments(comments) {
    const prepared = [];
    const summary = emptyMigrationSummary();
    comments.forEach((comment) => {
      if (!isCommentShape(comment)) {
        summary.checked += 1;
        summary.removed += 1;
        return;
      }
      if (comment.documentFingerprint === paper.fingerprint) {
        if (isValidComment(comment)) prepared.push(comment);
        else {
          summary.checked += 1;
          summary.removed += 1;
        }
        return;
      }

      summary.checked += 1;
      const migrated = migrateComment(comment);
      if (migrated) {
        prepared.push(migrated);
        summary.kept += 1;
      } else {
        summary.removed += 1;
      }
    });
    return { comments: prepared, summary };
  }

  function migrateComment(comment) {
    if (!migration
      || migration.toFingerprint !== paper.fingerprint
      || !Array.isArray(migration.pages)) return null;
    const anchor = comment.anchor?.candidates?.length
      ? comment.anchor
      : createTextAnchor(migration.pages, Number(comment.page), Number(comment.x), Number(comment.y));
    if (!anchor) return null;
    const location = locateTextAnchor(anchor);
    if (!location) return null;
    return {
      ...comment,
      page: location.page,
      x: round(location.x),
      y: round(location.y),
      anchor: createTextAnchor(paper.pages, location.page, location.x, location.y),
      documentFingerprint: paper.fingerprint,
    };
  }

  function createTextAnchor(pages, pageNumber, x, y) {
    const page = pages.find((candidate) => Number(candidate.number) === Number(pageNumber));
    if (!page?.words?.length) return null;
    const words = page.words
      .map((word, index) => ({ ...word, index, normalised: normaliseToken(word.text) }))
      .filter((word) => word.normalised);
    if (!words.length) return null;

    let nearest = null;
    words.forEach((word) => {
      const right = Number(word.x) + Number(word.w || 0);
      const bottom = Number(word.y) + Number(word.h || 0);
      const dx = Number(x) < Number(word.x) ? Number(word.x) - Number(x) : Number(x) > right ? Number(x) - right : 0;
      const dy = Number(y) < Number(word.y) ? Number(word.y) - Number(y) : Number(y) > bottom ? Number(y) - bottom : 0;
      const score = (dx * dx) + (dy * dy);
      if (!nearest || score < nearest.score) nearest = { word, score };
    });
    if (!nearest || nearest.score > 144) return null;

    const focusIndex = words.findIndex((word) => word.index === nearest.word.index);
    const candidates = [];
    [4, 3, 2, 1].forEach((radius) => {
      const start = Math.max(0, focusIndex - radius);
      const end = Math.min(words.length, focusIndex + radius + 1);
      const selected = words.slice(start, end);
      const text = selected.map((word) => word.normalised).join('');
      if (text.length < 10 || candidates.some((candidate) => candidate.text === text)) return;
      const focusOffset = selected
        .slice(0, focusIndex - start)
        .reduce((total, word) => total + word.normalised.length, 0)
        + Math.floor(nearest.word.normalised.length / 2);
      candidates.push({ text, focusOffset });
    });
    return candidates.length ? { candidates, page: Number(pageNumber), x: Number(x), y: Number(y) } : null;
  }

  function locateTextAnchor(anchor) {
    for (const candidate of anchor.candidates) {
      const matches = [];
      paper.pages.forEach((page) => {
        let joined = '';
        const ranges = [];
        page.words.forEach((word) => {
          const normalised = normaliseToken(word.text);
          if (!normalised) return;
          const start = joined.length;
          joined += normalised;
          ranges.push({ start, end: joined.length, word });
        });
        let matchStart = joined.indexOf(candidate.text);
        while (matchStart !== -1) {
          const focusCharacter = matchStart + candidate.focusOffset;
          const target = ranges.find((range) => focusCharacter >= range.start && focusCharacter < range.end);
          if (target) {
            const targetX = Number(target.word.x) + (Number(target.word.w || 0) / 2);
            const targetY = Number(target.word.y) + (Number(target.word.h || 0) / 2);
            const score = (Math.abs(Number(page.number) - Number(anchor.page)) * 100)
              + Math.abs(targetY - Number(anchor.y || 0));
            matches.push({ page: Number(page.number), x: targetX, y: targetY, score });
          }
          matchStart = joined.indexOf(candidate.text, matchStart + 1);
        }
      });
      if (matches.length) return matches.sort((a, b) => a.score - b.score)[0];
    }
    return null;
  }

  function normaliseToken(value) {
    return String(value || '')
      .normalize('NFKC')
      .toLocaleLowerCase('en')
      .replace(/[^\p{L}\p{N}]+/gu, '');
  }

  function exportComments() {
    const payload = {
      format: 'paper-review-comments',
      version: 2,
      document: paper.sourceFile,
      documentFingerprint: paper.fingerprint,
      exportedAt: new Date().toISOString(),
      comments: state.comments,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `data-centre-paper-comments-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast(`Exported ${state.comments.length} comment${state.comments.length === 1 ? '' : 's'}`);
  }

  async function importComments(event) {
    const [file] = event.target.files;
    event.target.value = '';
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const imported = Array.isArray(parsed) ? parsed : parsed.comments;
      if (!Array.isArray(imported) || !imported.every(isCommentShape)) throw new Error('Invalid comment file');
      const prepared = prepareComments(imported);
      const existing = new Map(state.comments.map((comment) => [comment.id, comment]));
      prepared.comments.forEach((comment) => existing.set(comment.id, comment));
      state.comments = [...existing.values()];
      persistComments();
      showList();
      renderAllComments();
      const removed = prepared.summary.removed ? `; ${prepared.summary.removed} obsolete removed` : '';
      showToast(`Imported ${prepared.comments.length} comment${prepared.comments.length === 1 ? '' : 's'}${removed}`);
    } catch {
      showToast('That file is not a valid paper review export');
    }
  }

  function isValidComment(comment) {
    return isCommentShape(comment)
      && Number(comment.page) >= 1
      && Number(comment.page) <= paper.pageCount;
  }

  function isCommentShape(comment) {
    return comment
      && typeof comment.id === 'string'
      && Number.isInteger(Number(comment.page))
      && Number.isFinite(Number(comment.x))
      && Number.isFinite(Number(comment.y))
      && typeof comment.body === 'string';
  }

  function changeZoom(direction) {
    const next = clamp(state.zoomIndex + direction, 0, ZOOM_LEVELS.length - 1);
    if (next === state.zoomIndex) return;
    state.zoomIndex = next;
    updateZoom();
  }

  function updateZoom() {
    const zoom = ZOOM_LEVELS[state.zoomIndex];
    document.documentElement.style.setProperty('--paper-width', `${BASE_PAGE_WIDTH * zoom}px`);
    els.zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
    els.zoomOutBtn.disabled = state.zoomIndex === 0;
    els.zoomInBtn.disabled = state.zoomIndex === ZOOM_LEVELS.length - 1;
  }

  function scrollToPage(pageNumber) {
    document.querySelector(`#page-${pageNumber}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function scrollToMarker(comment) {
    const marker = document.querySelector(`.comment-marker[data-id="${CSS.escape(comment.id)}"]`);
    marker?.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
  }

  function toggleSidebar() {
    if (window.matchMedia('(max-width: 940px)').matches) {
      document.body.classList.toggle('sidebar-open');
      els.toggleNotesBtn.setAttribute('aria-expanded', String(document.body.classList.contains('sidebar-open')));
    } else {
      document.body.classList.toggle('sidebar-collapsed');
      els.toggleNotesBtn.setAttribute('aria-expanded', String(!document.body.classList.contains('sidebar-collapsed')));
    }
  }

  function syncSidebarMode() {
    const isMobile = window.matchMedia('(max-width: 940px)').matches;
    if (isMobile) {
      document.body.classList.remove('sidebar-collapsed');
      els.toggleNotesBtn.setAttribute('aria-expanded', String(document.body.classList.contains('sidebar-open')));
    } else {
      document.body.classList.remove('sidebar-open');
      els.toggleNotesBtn.setAttribute('aria-expanded', String(!document.body.classList.contains('sidebar-collapsed')));
    }
  }

  function openSidebar() {
    if (window.matchMedia('(max-width: 940px)').matches) document.body.classList.add('sidebar-open');
    else document.body.classList.remove('sidebar-collapsed');
    els.toggleNotesBtn.setAttribute('aria-expanded', 'true');
  }

  function closeSidebar() {
    if (window.matchMedia('(max-width: 940px)').matches) document.body.classList.remove('sidebar-open');
    else document.body.classList.add('sidebar-collapsed');
    els.toggleNotesBtn.setAttribute('aria-expanded', 'false');
  }

  function showToast(message) {
    window.clearTimeout(state.toastTimer);
    els.toast.textContent = message;
    els.toast.hidden = false;
    state.toastTimer = window.setTimeout(() => { els.toast.hidden = true; }, 2600);
  }

  function createId() {
    return window.crypto?.randomUUID?.() || `comment-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function formatDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short' }).format(date);
  }

  function round(value) { return Math.round(value * 1000) / 1000; }
  function clamp(value, min, max) { return Math.min(Math.max(value, min), max); }
  function escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  }
})();

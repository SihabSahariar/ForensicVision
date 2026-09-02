/* ForensicVision project site - progressive enhancement only.
   Every section is readable and navigable with JavaScript disabled. */

(function () {
  "use strict";

  /* ------------------------------------------------------------- theme */

  var STORAGE_KEY = "fv-theme";
  var root = document.documentElement;

  function applyTheme(name) {
    if (name === "light") {
      root.setAttribute("data-theme", "light");
    } else {
      root.removeAttribute("data-theme");
    }
  }

  // The initial theme is set by an inline script in <head> to avoid a flash;
  // this only wires the toggle.
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      applyTheme(next);
      try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* private mode */ }
      toggle.setAttribute(
        "aria-label", next === "light" ? "Switch to dark theme" : "Switch to light theme"
      );
    });
  }

  /* -------------------------------------------------------- mobile nav */

  var navToggle = document.querySelector(".nav-toggle");
  var nav = document.querySelector(".site-nav");
  if (navToggle && nav) {
    navToggle.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.addEventListener("click", function (event) {
      if (event.target.tagName === "A") {
        nav.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  /* ------------------------------------------------------ copy buttons */

  document.querySelectorAll(".code").forEach(function (block) {
    var pre = block.querySelector("pre");
    var button = block.querySelector(".copy");
    if (!pre || !button) return;

    button.addEventListener("click", function () {
      var text = pre.innerText.replace(/ /g, " ");
      var done = function () {
        var original = button.textContent;
        button.textContent = "Copied";
        button.classList.add("is-done");
        setTimeout(function () {
          button.textContent = original;
          button.classList.remove("is-done");
        }, 1600);
      };

      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else {
        fallback();
      }

      function fallback() {
        // file:// and plain http have no async clipboard API.
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        try { document.execCommand("copy"); done(); } catch (e) { /* give up quietly */ }
        document.body.removeChild(area);
      }
    });
  });

  /* --------------------------------------------------------------- tabs */

  document.querySelectorAll("[data-tabs]").forEach(function (group) {
    var buttons = Array.prototype.slice.call(group.querySelectorAll(".tabs__btn"));
    var panels = Array.prototype.slice.call(group.querySelectorAll(".tabs__panel"));

    function select(index) {
      buttons.forEach(function (button, i) {
        button.setAttribute("aria-selected", i === index ? "true" : "false");
        button.tabIndex = i === index ? 0 : -1;
      });
      panels.forEach(function (panel, i) { panel.hidden = i !== index; });
    }

    buttons.forEach(function (button, index) {
      button.addEventListener("click", function () { select(index); });
      button.addEventListener("keydown", function (event) {
        var delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
        if (!delta) return;
        event.preventDefault();
        var next = (index + delta + buttons.length) % buttons.length;
        select(next);
        buttons[next].focus();
      });
    });

    select(0);
  });

  /* ----------------------------------------------------- gallery filter */

  var filters = document.querySelectorAll(".filter");
  var shots = Array.prototype.slice.call(document.querySelectorAll(".shot"));

  filters.forEach(function (filter) {
    filter.addEventListener("click", function () {
      var want = filter.dataset.filter;
      filters.forEach(function (other) { other.classList.toggle("is-active", other === filter); });
      shots.forEach(function (shot) {
        shot.hidden = want !== "all" && shot.dataset.group !== want;
      });
    });
  });

  /* ---------------------------------------------------------- lightbox */

  var lightbox = document.querySelector(".lightbox");
  if (lightbox && shots.length) {
    var image = lightbox.querySelector("img");
    var caption = lightbox.querySelector(".lightbox__cap");
    var current = 0;

    function visibleShots() {
      return shots.filter(function (shot) { return !shot.hidden; });
    }

    function show(index) {
      var list = visibleShots();
      if (!list.length) return;
      current = (index + list.length) % list.length;
      var shot = list[current];
      var thumb = shot.querySelector("img");
      image.src = thumb.getAttribute("src");
      image.alt = thumb.getAttribute("alt") || "";
      caption.textContent = shot.dataset.caption || "";
      lightbox.classList.add("is-open");
      document.body.style.overflow = "hidden";
    }

    function close() {
      lightbox.classList.remove("is-open");
      document.body.style.overflow = "";
      image.removeAttribute("src");
    }

    shots.forEach(function (shot) {
      shot.addEventListener("click", function () {
        show(visibleShots().indexOf(shot));
      });
      shot.setAttribute("tabindex", "0");
      shot.setAttribute("role", "button");
      shot.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          show(visibleShots().indexOf(shot));
        }
      });
    });

    lightbox.querySelector(".lightbox__close").addEventListener("click", close);
    lightbox.querySelector(".lightbox__prev").addEventListener("click", function (e) {
      e.stopPropagation();
      show(current - 1);
    });
    lightbox.querySelector(".lightbox__next").addEventListener("click", function (e) {
      e.stopPropagation();
      show(current + 1);
    });
    lightbox.addEventListener("click", function (event) {
      if (event.target === lightbox || event.target === image) close();
    });

    document.addEventListener("keydown", function (event) {
      if (!lightbox.classList.contains("is-open")) return;
      if (event.key === "Escape") close();
      if (event.key === "ArrowLeft") show(current - 1);
      if (event.key === "ArrowRight") show(current + 1);
    });
  }

  /* ---------------------------------------------------------- scrollspy */

  var spyLinks = Array.prototype.slice.call(
    document.querySelectorAll("[data-spy] a[href^='#']")
  );

  if (spyLinks.length && "IntersectionObserver" in window) {
    var targets = spyLinks
      .map(function (link) { return document.querySelector(link.getAttribute("href")); })
      .filter(Boolean);

    var seen = new Map();

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { seen.set(entry.target, entry); });

      var best = null;
      seen.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        if (!best || entry.target.offsetTop < best.target.offsetTop) best = entry;
      });
      if (!best) return;

      spyLinks.forEach(function (link) {
        link.classList.toggle(
          "is-active", link.getAttribute("href") === "#" + best.target.id
        );
      });
    }, { rootMargin: "-80px 0px -65% 0px", threshold: 0 });

    targets.forEach(function (target) { observer.observe(target); });
  }

  /* ------------------------------------------------------ heading links */

  document.querySelectorAll(".docs-body h2[id], .docs-body h3[id]").forEach(function (heading) {
    var link = document.createElement("a");
    link.className = "anchor";
    link.href = "#" + heading.id;
    link.textContent = "#";
    link.setAttribute("aria-label", "Link to this section");
    heading.appendChild(link);
  });

  /* ------------------------------------------------------ back to top */

  var toTop = document.querySelector(".to-top");
  if (toTop) {
    toTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    var onScroll = function () {
      toTop.classList.toggle("is-visible", window.scrollY > 700);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  /* ----------------------------------------------------- year in footer */

  document.querySelectorAll("[data-year]").forEach(function (node) {
    node.textContent = String(new Date().getFullYear());
  });
})();

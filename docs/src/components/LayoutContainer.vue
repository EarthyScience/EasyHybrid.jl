<!-- Adapted from https://github.com/withcatai/node-llama-cpp/blob/master/.vitepress/theme/LayoutContainer.vue -->

<script setup lang="ts">
import { provide, nextTick, onBeforeMount } from "vue";
import { useData, useRoute } from "vitepress";

const { isDark } = useData();
const route = useRoute();

const appShowAnimationName = "app-show";
const appShowAnimationDelay = 300;

/* View Transition Support */
interface ViewTransitionAPI {
  startViewTransition?: (
    callback: () => Promise<void> | void
  ) => { ready: Promise<void> };
}

const hasDocument = typeof document !== "undefined";

const themeTransitionEnabled =
  hasDocument &&
  (document as Document & ViewTransitionAPI).startViewTransition != null;

if (hasDocument) {
  document.documentElement.classList.toggle(
    "theme-transition",
    themeTransitionEnabled
  );
}


/* Helpers */
function getAppElement(): HTMLElement | null {
  return document.querySelector<HTMLElement>("#app");
}

function getAppShowAnimation(app: HTMLElement): CSSAnimation | undefined {
  return (app.getAnimations?.() ?? []).find(
    (animation): animation is CSSAnimation =>
      animation instanceof CSSAnimation &&
      animation.animationName === appShowAnimationName
  );
}

/* Initial Mount Animation Handling */
onBeforeMount(() => {
  if (!hasDocument) return;

  document.documentElement.classList.add("start-animation");

  const app = getAppElement();
  if (!app) return;

  const appShowAnimation = getAppShowAnimation(app);

  if (route.path === "/") {
    // Force homepage animation to re-trigger in production
    app.animate(
      { display: ["none", "initial"] },
      { duration: 1, easing: "linear" }
    );

    // Only cancel if the animation exists
    if (appShowAnimation) {
      appShowAnimation.cancel();
    }
  } else {
    // Only proceed if animation exists and currentTime is a number
    if (appShowAnimation) {
      const time = appShowAnimation.currentTime;
      if (typeof time === "number" && time < appShowAnimationDelay) {
        appShowAnimation.currentTime = appShowAnimationDelay;
      }
    }
  }

  setTimeout(() => {
    document.documentElement.classList.remove("start-animation");
  }, 2000);
});


/* Theme Toggle (View Transitions) */
provide("toggle-appearance", async () => {
  if (!hasDocument || !themeTransitionEnabled) {
    isDark.value = !isDark.value;
    return;
  }

  const showDark = !isDark.value;
  const prefersMotion = window.matchMedia(
    "(prefers-reduced-motion: no-preference)"
  ).matches;

  const doc = document as Document & ViewTransitionAPI;

  await doc.startViewTransition!(async () => {
    isDark.value = showDark;
    await nextTick();
  }).ready;

  document.documentElement.animate(
    prefersMotion
      ? {
          maskPosition: showDark
            ? ["0% 150%", "0% 75%"]
            : ["0% 25%", "0% -52%"]
        }
      : {
          opacity: showDark ? [0, 1] : [1, 0]
        },
    {
      duration: 300,
      easing: "ease-in-out",
      pseudoElement: showDark
        ? "::view-transition-new(root)"
        : "::view-transition-old(root)"
    }
  );
});
</script>

<template>
  <slot />
</template>

<style>
::view-transition-image-pair(root) {
  isolation: isolate;
}

::view-transition-old(root),
::view-transition-new(root) {
  animation: none;
  mix-blend-mode: normal;
  display: block;
}

.dark::view-transition-old(root),
::view-transition-new(root) {
  z-index: 1;
}

::view-transition-old(root),
.dark::view-transition-new(root) {
  mask: linear-gradient(
      to bottom,
      rgb(0 0 0 / 0%) 0%,
      black calc((50 / 300) * 100%),
      black calc((1 - (50 / 300)) * 100%),
      rgb(0 0 0 / 0%) 100%
    )
    content-box 0 75% / 100% 300% no-repeat;
  z-index: 9999;
}

html.theme-transition .VPSwitch.VPSwitchAppearance > .check {
  transition-duration: 0s !important;
}
</style>

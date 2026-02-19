// .vitepress/theme/index.ts

import "./smoothLoad.css"; // smooth loading on first visit

import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'
import 'virtual:mathjax-styles.css'

import { 
  NolebaseEnhancedReadabilitiesMenu, 
  NolebaseEnhancedReadabilitiesScreenMenu, 
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from "@/VersionPicker.vue"
import StarUs from '@/StarUs.vue'
import AuthorBadge from '@/AuthorBadge.vue'
import Authors from '@/Authors.vue'

// light/dark theme toggle with view transitions
import LayoutContainer from '@/LayoutContainer.vue' 

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css'
import './docstrings.css'

export const Theme: ThemeConfig = {
  extends: DefaultTheme,

  // Use LayoutContainer for the main layout
  Layout() {
    return h(LayoutContainer, null, {
      // Inside LayoutContainer we can still render DefaultTheme.Layout for default slots
      default: () => h(DefaultTheme.Layout, null, {
        'nav-bar-content-after': () => [
          h(StarUs),
          h(NolebaseEnhancedReadabilitiesMenu), // Enhanced Readabilities menu
        ],
        'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
      }),
    })
  },
  enhanceApp({ app, router, siteData }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
    app.component('AuthorBadge', AuthorBadge)
    app.component('Authors', Authors)
  }
}
export default Theme
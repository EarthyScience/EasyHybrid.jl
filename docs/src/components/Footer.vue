
<script setup>
import { withBase } from 'vitepress'
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}
</script>

<template>
  <footer class="custom-footer">
    <div class="footer-top">
      <div class="footer-col">
        <h4>EXPLORE</h4>
        <ul>
          <li><a href="/get_started">Get Started</a></li>
          <li><a href="tutorials/overview">Tutorials</a></li>
          <li><a href="/research/overview">Research</a></li>
          <!-- <li><a href="/libraries">Libraries</a></li> -->
          <!-- <li><a href="/news">News</a></li> -->
          <!-- <li><a href="/team">Team</a></li> -->
        </ul>
      </div>

      <div class="footer-col">
        <h4>CONNECT</h4>
        <ul class="connect-list">
          <li>
            <a href="https://github.com/EarthyScience/EasyHybrid.jl" target="_blank">
              <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.38.6.1.82-.26.82-.58v-2.03c-3.34.72-4.04-1.6-4.04-1.6-.55-1.4-1.34-1.77-1.34-1.77-1.08-.74.08-.72.08-.72 1.2.08 1.83 1.23 1.83 1.23 1.07 1.83 2.8 1.3 3.48 1 .1-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02 0 2.04.13 3 .4 2.28-1.55 3.28-1.23 3.28-1.23.65 1.66.24 2.88.12 3.18.77.84 1.23 1.9 1.23 3.22 0 4.61-2.8 5.63-5.48 5.92.43.37.82 1.1.82 2.22v3.29c0 .32.22.7.83.58C20.57 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z"/></svg>
              GitHub
            </a>
          </li>
          <li>
            <a href="https://github.com/EarthyScience/EasyHybrid.jl/discussions" target="_blank">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 11.5c0 4.7-4 8.5-9 8.5-1.3 0-2.5-.2-3.7-.6L3 21l1.7-4C3.6 15.6 3 13.7 3 11.5 3 6.8 7 3 12 3s9 3.8 9 8.5z"/>                
                <circle cx="8.5" cy="11.5" r="0.2"/>
                <circle cx="12" cy="11.5" r="0.2"/>
                <circle cx="15.5" cy="11.5" r="0.2"/>
              </svg>
              Discussions
            </a>
          </li>
          <li>
            <a href="https://www.bgc-jena.mpg.de/en/bgi/miss" target="_blank">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <line x1="4" y1="10" x2="20" y2="10"/>
                <line x1="4" y1="14" x2="20" y2="14"/>
                <line x1="4" y1="18" x2="20" y2="18"/>

                <circle cx="9" cy="12" r="1.5"/>
                <circle cx="15" cy="16" r="1.5"/>
              </svg>
              MISS
            </a>
          </li>
        </ul>
      </div>

      <div class="footer-col footer-supported">
        <h4><strong>SUPPORTED BY LEADING RESEARCHERS</strong></h4>
        <p>EasyHybrid.jl is developed by researchers at the</p>
        <div class="logo-row">
          <a href="https://www.bgc-jena.mpg.de/en/bgi/home" target="_blank">
            <img :src="withBase('MPI_BGC_wide_E_green_cmyk.png')" alt="MPI-BGC-light-mode" />
          </a>
          <a href="https://www.bgc-jena.mpg.de/en/bgi/home" target="_blank">
            <img :src="withBase('MPI_BGC_wide_E_neg_cmyk.png')" alt="MPI-BGC-dark-mode" />
          </a>
        </div>
      </div>
    </div>

    <div class="footer-bottom">
      <span>
        © 2023 The EasyHybrid Project Contributors.
      </span>
      <button class="back-to-top" @click="scrollToTop">↑ Back to top</button>
    </div>
  </footer>
</template>

<style scoped>
.custom-footer {
  position: relative;
  /* margin-top: -64px; */
  bottom: 0;
  border-top: 1px solid var(--vp-c-divider);
  padding: 48px 48px 24px;
  margin-left: -31px;
  margin-right: -31px;
  padding-left: calc(31px + 31px);
  padding-right: calc(31px + 31px);
  background: var(--vp-sidebar-bg-color);
  font-size: 14px;
  color: var(--vp-c-text-2);
  z-index: 30;
}


.footer-top {
  display: grid;
  grid-template-columns: 1fr 1fr 2fr;
  gap: 48px;
  margin-bottom: 48px;
  max-width: 1152px;
  margin-left: auto;
  margin-right: auto;
}

h4 {
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--vp-c-text-1);
  margin-bottom: 16px;
}

ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

a {
  color: var(--vp-c-text-2);
  text-decoration: none;
  transition: color 0.2s;
}

a:hover {
  color: var(--vp-c-text-1);
}

.connect-list a {
  display: flex;
  align-items: center;
  gap: 10px;
}

.icon {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
}

.footer-supported p {
  margin: 0 0 20px;
  line-height: 1.6;
}

.logo-row {
  display: flex;
  align-items: center;
  gap: 32px;
  flex-wrap: wrap;
}

.logo-row img {
  height: 48px;
  width: auto;
  filter: var(--vp-logo-filter, none);
}
.footer-bottom {
  border-top: 1px solid var(--vp-c-divider);
  padding-top: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  max-width: 1152px;
  margin-left: auto;
  margin-right: auto;
}
.footer-bottom > span {
  flex: 1;
  line-height: 1.7;
}

.footer-bottom a {
  color: var(--vp-c-brand-1);
}

.back-to-top {
  padding: 6px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  font-size: 13px;
  transition: border-color 0.2s, color 0.2s;
  white-space: nowrap;
}

.back-to-top:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.website-source {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--vp-c-brand-1) !important;
  white-space: nowrap;
}

@media (max-width: 1024px) {
  .custom-footer {
    margin-top: 24px;
  }
  .footer-top {
    grid-template-columns: 1fr 1fr;
  }

  .footer-supported {
    grid-column: 1 / -1;
  }
}

@media (max-width: 768px) {
  .custom-footer {
    padding: 32px 24px 16px;
  }

  .footer-top {
    grid-template-columns: 1fr 1fr;
    gap: 32px;
  }

  .footer-supported {
    grid-column: 1 / -1;
  }

  .footer-bottom {
    flex-direction: column;
    align-items: center;
    gap: 12px;
    text-align: center;
  }

  .footer-bottom > span {
    width: 100%;
    text-align: center;
  }

  .back-to-top {
    width: 100%;
    text-align: center;
  }

  .website-source {
    align-self: center;
  }
}

@media (max-width: 768px) {
  .custom-footer {
    margin-left: -24px;
    margin-right: -24px;
    padding-left: calc(24px + 24px);
    padding-right: calc(24px + 24px);
  }
}

@media (max-width: 480px) {
  .footer-top {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 28px;
  }

  .footer-col {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .footer-supported {
    grid-column: unset;
  }

  ul {
    align-items: center;
  }

  .connect-list a {
    justify-content: center;
  }

  .logo-row {
    justify-content: center;
  }
}
</style>
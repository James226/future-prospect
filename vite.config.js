import { defineConfig } from 'vite'

export default defineConfig({
  root: 'src/renderer',
  test: {
    browser: {
      enabled: true,
      name: 'chrome'
    }
  }
})

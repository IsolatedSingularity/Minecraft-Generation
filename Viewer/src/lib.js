import * as THREE from "three"
import * as renderer from "../vendor/block-model-renderer/browser.js"

let configured = false

export function loadLibrary() {
  if (!configured) {
    renderer.configure({
      three: THREE,
      assetsUrl: new URL("../vendor/block-model-renderer/assets.zip", import.meta.url)
    })
    configured = true
  }
  return Promise.resolve(renderer)
}

export { THREE }

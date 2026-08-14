import * as THREE from "three"
import * as renderer from "../vendor/block-model-renderer/browser.js"
import rendererAssetsUrl from "../vendor/block-model-renderer/assets.zip?url"

let configured = false

export function loadLibrary() {
  if (!configured) {
    renderer.configure({
      three: THREE,
      assetsUrl: rendererAssetsUrl
    })
    configured = true
  }
  return Promise.resolve(renderer)
}

export { THREE }

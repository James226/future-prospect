export interface QueueItem {
  items: GPUCommandBuffer[]
  callback: () => Promise<void>
}

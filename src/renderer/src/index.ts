import Stats from 'stats.js'
import Renderer from './renderer'
import Game from './game'
import { mat4 } from 'gl-matrix'
;(async function (): Promise<void> {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement

  const projectionMatrix = mat4.create()

  const steamworks: typeof import('steamworks.js') = eval(
    'typeof require === "function" && require(\'steamworks.js\')'
  )

  if (steamworks) {
    try {
      const client = steamworks.init(2377570)

      console.log(client.localplayer.getName())
      client.localplayer.getSteamId()

      if (!client.achievement.activate('ACH_WIN_ONE_GAME')) {
        console.log('Sad fish')
      }

      // client.workshop.createItem()
      //   .then(async item => {
      //     const update: UgcUpdate = {};
      //     update.title = "Test";
      //     await client.workshop.updateItem(item.itemId, update);
      //     client.workshop.
      //     console.log(item);
      //   });

      client.auth.getSessionTicket().then((ticket) => {
        const buffer = ticket.getBytes()
        console.log('ticket', buffer.toString('hex'))
      })

      // client.matchmaking.getLobbies().then(async (lobbies) => {
      //   console.log('Lobbies', lobbies)
      //   const lobby = await lobbies[0].join()
      //   console.log(lobby.getMembers())
      //
      //   lobby.leave()
      // })
    } catch {
      /* empty */
    }
  }

  const renderer = await Renderer.init(canvas)
  const configureRenderer: () => void = () => {
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight

    const aspect = Math.abs(canvas.width / canvas.height)
    mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 1, 100000000)

    renderer.configure(canvas.width, canvas.height)
  }

  window.addEventListener('resize', configureRenderer, false)
  window.addEventListener('click', () => {
    canvas.requestPointerLock()
  })

  let game = await Game.init(renderer.device)

  configureRenderer()

  let lastUpdate = performance.now()

  const stats = new Stats()
  stats.showPanel(0)
  document.body.appendChild(stats.dom)

  if (import.meta.hot) {
    import.meta.hot.accept(['./game.ts'], async () => {
      game = await Game.init(renderer.device)
    })
  }

  window.addEventListener('beforeunload', () => {
    renderer.running = false
    game.destroy()
  })

  const doFrame: (number) => Promise<void> = async (timestamp: number) => {
    if (!renderer.running) return

    stats.begin()
    await game.update(renderer.device, projectionMatrix, timestamp)
    renderer.render((e) => game.draw(e))
    const now = performance.now()

    if (now - lastUpdate > 1000) {
      lastUpdate = now
    }

    stats.end()
    requestAnimationFrame(doFrame)
  }
  requestAnimationFrame(doFrame)
})()

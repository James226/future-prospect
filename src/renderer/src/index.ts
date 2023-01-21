import Stats from 'stats.js'
import Renderer from './renderer'
import Game from './game'
import { mat4 } from 'gl-matrix'

const canvas = document.getElementById('canvas') as HTMLCanvasElement

const projectionMatrix = mat4.create()

const steamworks: typeof import('steamworks.js') = eval(
  'typeof require === "function" && require(\'steamworks.js\')'
)

if (steamworks) {
  const client = steamworks.init(480)

  // Print Steam username
  console.log(client.localplayer.getName())
  client.localplayer.getSteamId()

  // Tries to activate an achievement
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

  client.matchmaking.getLobbies().then(async (lobbies) => {
    console.log('Lobbies', lobbies)
    const lobby = await lobbies[0].join()
    console.log(lobby.getMembers())

    lobby.leave()
  })
}

Renderer.init(canvas).then(async (renderer) => {
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

  const game = await Game.init(renderer.device)

  configureRenderer()

  let lastUpdate = performance.now()

  const stats = new Stats()
  stats.showPanel(0)
  document.body.appendChild(stats.dom)

  const doFrame: (number) => void = (timestamp: number) => {
    stats.begin()
    game.update(renderer.device, projectionMatrix, timestamp).then(() => {
      renderer.render((e) => game.draw(e))
      const now = performance.now()

      if (now - lastUpdate > 1000) {
        lastUpdate = now
      }

      stats.end()
      requestAnimationFrame(doFrame)
    })
  }
  requestAnimationFrame(doFrame)
})

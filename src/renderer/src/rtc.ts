class RealTimeCommunication {
  socket: RTCPeerConnection
  channel: undefined | RTCDataChannel = undefined
  onMessage: (message: MessageEvent) => void

  constructor(
    onIceCandidate: (candidate: RTCIceCandidate) => void,
    onMessage: (message: MessageEvent) => void
  ) {
    this.onMessage = onMessage
    console.log('Creating connection')
    this.socket = new RTCPeerConnection({
      iceServers: [
        {
          urls: 'stun:stun2.l.google.com:19302'
        },
        {
          urls: 'turn:relay.backups.cz',
          credential: 'webrtc',
          username: 'webrtc'
        }
      ]
    })

    this.socket.ondatachannel = (e): void => {
      console.log('Channel received: ', e)
      this.channel = e.channel
      this.channel.onmessage = (message): void => {
        this.onMessage(message)
      }
      this.channel.onopen = (e): void => {
        console.log('Channel opened', e)
      }
      this.channel.onclose = (e): void => {
        console.log('Channel closed', e)
      }
    }

    this.socket.onicecandidate = (event): void => {
      if (event.candidate) {
        onIceCandidate(event.candidate)
        console.log('Candidate', JSON.stringify(event.candidate))
      }
    }
  }

  addIce(ice: RTCIceCandidateInit | RTCIceCandidate): void {
    this.socket
      .addIceCandidate(ice)
      .then(() => console.log('Success'))
      .catch((e) => console.log('Error', e))
  }

  offer(): Promise<RTCSessionDescriptionInit> {
    return new Promise((resolve) => {
      this.channel = this.socket.createDataChannel('sendDataChannel', undefined)

      this.channel.onopen = (): void => console.log('Open: ', this.channel!.readyState)
      this.channel.onclose = (): void => console.log('Closed: ', this.channel!.readyState)
      this.channel.onmessage = (message): void => {
        this.onMessage(message)
      }

      this.socket.createOffer().then(
        (desc) => {
          this.socket.setLocalDescription(desc)
          console.log('Offer from localConnection', desc.sdp)
          console.log(JSON.stringify(desc))
          resolve(desc)
        },
        (error) => {
          console.log('Failed to create session description: ', error.toString())
        }
      )
    })
  }

  answer(desc: RTCSessionDescriptionInit): Promise<void | RTCSessionDescriptionInit> {
    this.socket.setRemoteDescription(desc)
    return this.socket.createAnswer().then(
      (desc) => {
        this.socket.setLocalDescription(desc)
        console.log(JSON.stringify(desc))
        return desc
      },
      (e) => {
        console.log('Answer failed', e)
      }
    )
  }

  complete(desc: RTCSessionDescriptionInit): void {
    this.socket.setRemoteDescription(desc)
  }

  send(message: string): void {
    if (this.channel && this.channel.readyState == 'open') {
      this.channel.send(message)
    }
  }
}

export default RealTimeCommunication

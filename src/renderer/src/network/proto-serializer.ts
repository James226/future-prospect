import protobuf, { Type } from 'protobufjs'

export default class ProtoSerializer {
  constructor(private message: Type) {}

  static async init(): Promise<ProtoSerializer> {
    const root = await protobuf.load('assets/message.proto')
    const message = root.lookupType('Message')

    return new ProtoSerializer(message)
  }

  serialize(payload: { [k: string]: unknown }): Uint8Array {
    const errMsg = this.message.verify(payload)
    if (errMsg) throw Error(errMsg)
    return this.message.encode(this.message.create(payload)).finish()
  }

  deserialize(buffer: Uint8Array): unknown {
    const buf = this.message.decode(buffer)
    return this.message.toObject(buf, {
      longs: String,
      enums: String,
      bytes: String
    })
  }
}

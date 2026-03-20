# Alerts — Pub/Sub

```
AlertPublisher  ──publish(msg)──►  AlertSubscriber.on_alert(msg)
                                   AlertSubscriber.on_alert(msg)
                                   ...
```

`AlertPublisher` is the bus. Subscribers register with `subscribe(sub, min_level)`.
Each subscriber only receives messages at or above its `min_level`.
Global cooldown deduplicates repeated alerts by title.

## Adding a new channel

```python
class TelegramBot(AlertSubscriber):
    async def on_alert(self, message: AlertMessage) -> bool:
        # send to Telegram
        return True

    async def test_connection(self) -> bool:
        return True

publisher.subscribe(TelegramBot(...), min_level=AlertLevel.INFO)
```

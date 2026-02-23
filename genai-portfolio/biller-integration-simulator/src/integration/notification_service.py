"""
Event notification service.

Handles outbound notifications triggered by payment lifecycle events —
payment confirmations, failures, settlement completions, and autopay
reminders. Channels include email, SMS, and push notifications.

In production, this would integrate with SendGrid, Twilio, or Firebase.
Here we simulate the notification dispatch and maintain an audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from src.models.biller import Biller
from src.models.payment import Payment, PaymentStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


class NotificationEvent(str, Enum):
    PAYMENT_CONFIRMATION = "payment_confirmation"
    PAYMENT_FAILURE = "payment_failure"
    SETTLEMENT_COMPLETE = "settlement_complete"
    AUTOPAY_REMINDER = "autopay_reminder"
    AUTOPAY_PROCESSED = "autopay_processed"
    EXCEPTION_ESCALATION = "exception_escalation"
    BILLER_ONBOARDED = "biller_onboarded"


@dataclass
class Notification:
    """A single notification to be dispatched."""
    notification_id: str
    event: NotificationEvent
    channel: NotificationChannel
    recipient: str
    subject: str
    body: str
    biller_id: Optional[str] = None
    transaction_id: Optional[str] = None
    dispatched: bool = False
    dispatched_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "notification_id": self.notification_id,
            "event": self.event.value,
            "channel": self.channel.value,
            "recipient": self.recipient,
            "subject": self.subject,
            "dispatched": self.dispatched,
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
        }


class NotificationService:
    """
    Dispatches notifications based on biller configuration and payment events.

    Respects the biller's notification preferences (which events to notify on,
    which channels to use) and maintains a complete audit log of all dispatched
    notifications.
    """

    def __init__(self):
        self._notifications: list[Notification] = []
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"NOTIF-{self._counter:06d}"

    def notify_payment_confirmation(
        self,
        payment: Payment,
        biller: Biller,
        recipient_email: Optional[str] = None,
    ) -> list[Notification]:
        """Send payment confirmation notification(s)."""
        if not biller.notifications.payment_confirmation:
            return []

        notifications: list[Notification] = []

        for channel_name in biller.notifications.channels:
            channel = NotificationChannel(channel_name)
            recipient = recipient_email or "customer@example.com"

            notification = Notification(
                notification_id=self._next_id(),
                event=NotificationEvent.PAYMENT_CONFIRMATION,
                channel=channel,
                recipient=recipient if channel == NotificationChannel.EMAIL else "***-***-0000",
                subject=f"Payment Confirmation - {biller.biller_name}",
                body=(
                    f"Your payment of ${payment.amount} to {biller.biller_name} "
                    f"has been received and processed. "
                    f"Transaction ID: {payment.transaction_id}. "
                    f"Account: ****{payment.customer_account_id[-4:]}."
                ),
                biller_id=biller.biller_id,
                transaction_id=payment.transaction_id,
            )

            self._dispatch(notification)
            notifications.append(notification)

        return notifications

    def notify_payment_failure(
        self,
        payment: Payment,
        biller: Biller,
        reason: str = "",
        recipient_email: Optional[str] = None,
    ) -> list[Notification]:
        """Send payment failure notification(s)."""
        if not biller.notifications.payment_failure:
            return []

        notifications: list[Notification] = []

        for channel_name in biller.notifications.channels:
            channel = NotificationChannel(channel_name)
            recipient = recipient_email or "customer@example.com"

            notification = Notification(
                notification_id=self._next_id(),
                event=NotificationEvent.PAYMENT_FAILURE,
                channel=channel,
                recipient=recipient if channel == NotificationChannel.EMAIL else "***-***-0000",
                subject=f"Payment Failed - {biller.biller_name}",
                body=(
                    f"Your payment of ${payment.amount} to {biller.biller_name} "
                    f"could not be processed. {reason} "
                    f"Please try again or contact support. "
                    f"Reference: {payment.transaction_id}."
                ),
                biller_id=biller.biller_id,
                transaction_id=payment.transaction_id,
            )

            self._dispatch(notification)
            notifications.append(notification)

        return notifications

    def notify_settlement_complete(
        self,
        biller: Biller,
        batch_id: str,
        total_amount: str,
        record_count: int,
    ) -> list[Notification]:
        """Notify biller that a settlement batch has been completed."""
        if not biller.notifications.settlement_complete:
            return []

        notifications: list[Notification] = []

        notification = Notification(
            notification_id=self._next_id(),
            event=NotificationEvent.SETTLEMENT_COMPLETE,
            channel=NotificationChannel.EMAIL,
            recipient=f"settlements@{biller.biller_name.lower().replace(' ', '')}.com",
            subject=f"Settlement Complete - Batch {batch_id}",
            body=(
                f"Settlement batch {batch_id} has been completed for "
                f"{biller.biller_name}. "
                f"Total settled: ${total_amount} across {record_count} transactions."
            ),
            biller_id=biller.biller_id,
        )

        self._dispatch(notification)
        notifications.append(notification)

        return notifications

    def notify_escalation(
        self,
        case_id: str,
        severity: str,
        role: str,
        description: str,
    ) -> Notification:
        """Send an exception escalation notification."""
        notification = Notification(
            notification_id=self._next_id(),
            event=NotificationEvent.EXCEPTION_ESCALATION,
            channel=NotificationChannel.EMAIL,
            recipient=f"{role}@operations.internal",
            subject=f"[{severity.upper()}] Exception Escalation - {case_id}",
            body=(
                f"Exception case {case_id} requires attention.\n"
                f"Severity: {severity}\n"
                f"Description: {description}\n"
                f"Please review and take action."
            ),
        )

        self._dispatch(notification)
        return notification

    def get_notifications(
        self,
        event: Optional[NotificationEvent] = None,
    ) -> list[Notification]:
        """Retrieve notifications, optionally filtered by event type."""
        if event:
            return [n for n in self._notifications if n.event == event]
        return list(self._notifications)

    def get_stats(self) -> dict:
        by_event: dict[str, int] = {}
        for n in self._notifications:
            key = n.event.value
            by_event[key] = by_event.get(key, 0) + 1

        by_channel: dict[str, int] = {}
        for n in self._notifications:
            key = n.channel.value
            by_channel[key] = by_channel.get(key, 0) + 1

        return {
            "total_dispatched": len(self._notifications),
            "by_event": by_event,
            "by_channel": by_channel,
        }

    def _dispatch(self, notification: Notification) -> None:
        """Simulate dispatching a notification."""
        notification.dispatched = True
        notification.dispatched_at = datetime.utcnow()
        self._notifications.append(notification)

        logger.info(
            f"Notification dispatched: {notification.notification_id}",
            extra={"event_data": notification.to_dict()},
        )

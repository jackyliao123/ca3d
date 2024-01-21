pub enum UserEvent {
    RequestCursorLock(bool),
    NotifyCursorLockStatus(bool),
    RequestResize,
}

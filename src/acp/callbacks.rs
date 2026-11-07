//! Internal stable-v1 ACP callback bundle.
#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "exposed only after prompt and service wiring are complete"
    )
)]

use std::sync::Arc;

use agent_client_protocol::v1::schema::{
    CancelNotification, InitializeRequest, LoadSessionRequest, NewSessionRequest,
};
use agent_client_protocol::{Agent, Client, ConnectTo, Error as ProtocolError};

use super::error::ControlError;
use super::session_control::SessionControl;

/// Build the unexposed stable-v1 callbacks over the shared session control boundary.
pub(crate) fn callback_bundle(control: Arc<SessionControl>) -> impl ConnectTo<Client> {
    let new_control = Arc::clone(&control);
    let load_control = Arc::clone(&control);
    let cancel_control = control;

    Agent
        .v1()
        .name("icode")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _connection| {
                responder.respond(SessionControl::initialize(&request))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: NewSessionRequest, responder, _connection| {
                let control = Arc::clone(&new_control);
                let result = tokio::task::spawn_blocking(move || control.new_session(&request))
                    .await
                    .map_err(ControlError::internal)
                    .and_then(|result| result);
                responder.respond_with_result(result.map_err(ProtocolError::from))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: LoadSessionRequest, responder, connection| {
                let control = Arc::clone(&load_control);
                let transaction =
                    tokio::task::spawn_blocking(move || control.prepare_load(&request))
                        .await
                        .map_err(ControlError::internal)
                        .and_then(|result| result);
                let transaction = match transaction {
                    Ok(transaction) => transaction,
                    Err(error) => return responder.respond_with_error(error.into()),
                };
                transaction
                    .complete_with(
                        |notification| connection.send_notification(notification),
                        |response| responder.respond(response),
                    )
                    .map_err(ProtocolError::from)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: CancelNotification,
                        _connection: agent_client_protocol::v1::ConnectionTo<Client>| {
                let _matched_active_prompt = cancel_control.cancel(&notification);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use agent_client_protocol::schema::ProtocolVersion;
    use agent_client_protocol::v1::schema::{
        AgentCapabilities, InitializeResponse, LoadSessionResponse, NewSessionResponse, SessionId,
        SessionNotification,
    };
    use agent_client_protocol::ErrorCode;
    use tempfile::TempDir;

    use super::*;
    use crate::acp::sessions::SessionRegistry;
    use crate::session::{Record, SessionStore};

    #[tokio::test(flavor = "current_thread")]
    async fn callbacks_preserve_load_order_errors_and_cancellation() -> Result<(), ProtocolError> {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let sessions = Arc::new(SessionRegistry::new(store.clone()));
        let control = Arc::new(SessionControl::new(
            Arc::clone(&sessions),
            "model-a",
            "build",
        ));

        let loaded = store
            .create(directory.path().to_str().unwrap(), "model-a", "build")
            .unwrap();
        loaded
            .append(&Record::User {
                text: "question".to_string(),
            })
            .unwrap();
        loaded
            .append(&Record::Assistant {
                content: Some("answer".to_string()),
                reasoning: None,
                tool_calls: vec![],
                responses_items: vec![],
            })
            .unwrap();
        let load_id = SessionId::new(loaded.id());

        let cancel_id = sessions
            .create(directory.path().to_str().unwrap(), "model-a", "build")
            .unwrap();
        let mut prompt = sessions.begin_prompt(&cancel_id).unwrap();
        let notifications = Arc::new(Mutex::new(Vec::new()));
        let received = Arc::clone(&notifications);
        let client = Client.v1().on_receive_notification(
            async move |notification: SessionNotification, _connection| {
                received
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(notification.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        );

        client
            .connect_with(callback_bundle(control), async move |connection| {
                let initialized: InitializeResponse = connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                assert_eq!(initialized.protocol_version, ProtocolVersion::V1);
                assert_eq!(initialized.agent_capabilities, AgentCapabilities::new());

                let invalid = connection
                    .send_request(NewSessionRequest::new("relative"))
                    .block_task()
                    .await
                    .unwrap_err();
                assert_eq!(invalid.code, ErrorCode::INVALID_PARAMS);

                let created: NewSessionResponse = connection
                    .send_request(NewSessionRequest::new(directory.path()))
                    .block_task()
                    .await?;
                assert!(store.load(&created.session_id.to_string()).is_ok());

                let _: LoadSessionResponse = connection
                    .send_request(LoadSessionRequest::new(load_id, directory.path()))
                    .block_task()
                    .await?;
                {
                    let updates = notifications
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    assert_eq!(updates.len(), 2);
                    assert_eq!(
                        serde_json::to_value(&updates[0]).unwrap()["sessionUpdate"],
                        "user_message_chunk"
                    );
                    assert_eq!(
                        serde_json::to_value(&updates[1]).unwrap()["sessionUpdate"],
                        "agent_message_chunk"
                    );
                }

                connection.send_notification(CancelNotification::new(cancel_id))?;
                tokio::time::timeout(Duration::from_secs(1), prompt.cancelled())
                    .await
                    .expect("cancel callback did not signal the active prompt");
                Ok(())
            })
            .await
    }

    #[test]
    fn callback_bundle_has_the_stable_v1_component_type() {
        fn accepts_component(_component: impl ConnectTo<Client>) {}

        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let sessions = Arc::new(SessionRegistry::new(store));
        accepts_component(callback_bundle(Arc::new(SessionControl::new(
            sessions, "model-a", "build",
        ))));
    }
}

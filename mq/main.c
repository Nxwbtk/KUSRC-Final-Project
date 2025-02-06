#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <amqp.h>
#include <amqp_tcp_socket.h>

#define HOSTNAME "localhost"
#define PORT 5672
#define USERNAME "nxwbtk"
#define PASSWORD "password"
#define QUEUE_NAME "hello"

void die_on_error(int x, char const *context) {
    if (x < 0) {
        fprintf(stderr, "%s: %s\n", context, amqp_error_string2(x));
        exit(1);
    }
}

void die_on_amqp_error(amqp_rpc_reply_t x, char const *context) {
    if (x.reply_type != AMQP_RESPONSE_NORMAL) {
        fprintf(stderr, "%s: AMQP error\n", context);
        exit(1);
    }
}

int main() {
    amqp_connection_state_t conn;
    amqp_socket_t *socket = NULL;
    amqp_rpc_reply_t res;
    int i = 0;

    conn = amqp_new_connection();
    socket = amqp_tcp_socket_new(conn);
    if (!socket) {
        fprintf(stderr, "Error creating TCP socket\n");
        return 1;
    }

    die_on_error(amqp_socket_open(socket, HOSTNAME, PORT), "Opening TCP socket");
    die_on_amqp_error(amqp_login(conn, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN, USERNAME, PASSWORD), "Logging in");
    amqp_channel_open(conn, 1);
    die_on_amqp_error(amqp_get_rpc_reply(conn), "Opening channel");

    amqp_basic_consume(conn, 1, amqp_cstring_bytes(QUEUE_NAME), amqp_empty_bytes, 0, 1, 0, amqp_empty_table);
    die_on_amqp_error(amqp_get_rpc_reply(conn), "Consuming");

    printf("Waiting for messages...\n");

    while (1) {
        amqp_envelope_t envelope;
        amqp_maybe_release_buffers(conn);

        res = amqp_consume_message(conn, &envelope, NULL, 0);
        if (res.reply_type == AMQP_RESPONSE_NORMAL) {
            printf("#%d Received message: %.*s\n", i,(int)envelope.message.body.len, (char *)envelope.message.body.bytes);
            i++;
            amqp_destroy_envelope(&envelope);
        } else {
            break;
        }
    }

    amqp_channel_close(conn, 1, AMQP_REPLY_SUCCESS);
    amqp_connection_close(conn, AMQP_REPLY_SUCCESS);
    amqp_destroy_connection(conn);
    
    return 0;
}

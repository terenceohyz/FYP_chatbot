async function fetchSessionId() {
    console.log("Fetching session ID...");
    try {
        const response = await fetch('http://localhost:8001/api/get-session-id');
        console.log("Response received from /api/get-session-id");
        if (!response.ok) throw new Error('Failed to fetch session ID');
        const data = await response.json();
        const sessionId = data.session_id;
        console.log("Session ID fetched:", sessionId);
        return sessionId;
    } catch (error) {
        console.error("Error fetching session ID:", error);
        return null;
    }
}

async function fetchEncryptionKey() {
    try {
        const response = await fetch('http://localhost:8001/api/get-key', {
            headers: {
                'Authorization': 'Bearer abc'
            }
        });
        if (!response.ok) throw new Error('Failed to fetch the key');
        const data = await response.json();
        return data.key;
    } catch (error) {
        console.error("Error fetching encryption key:", error);
        return null;
    }
}

async function setupWebSocket(sessionId, key) {
    const wsUrl = `ws://localhost:8001/ws/${sessionId}`;
    
    try {
        let socket = new WebSocket(wsUrl);

        socket.onopen = function(e) {
            console.log("Connection established! Ready to send messages.");
        };

        socket.onmessage = async function(event) {
            console.log("Message from server: ", event.data);
            try {
                const decryptedMessage = await decryptData(event.data, key);
                console.log("Decrypted Message: ", decryptedMessage);
            } catch (err) {
                console.error("Error decrypting message: ", err);
            }
        };

        socket.onclose = function(event) {
            if (event.wasClean) {
                console.log(`Connection closed cleanly, code=${event.code}, reason=${event.reason}`);
            } else {
                console.log('Connection died');
            }
        };

        socket.onerror = function(error) {
            console.error(`[WebSocket Error] ${error.message}`);
        };
    } catch (error) {
        console.error("WebSocket initialization error:", error);
    }
}

async function initializeWebSocket() {
    try {
        const sessionId = await fetchSessionId();
        if (sessionId) {
            const key = await fetchEncryptionKey();
            if (key) {
                await setupWebSocket(sessionId, key);
            } else {
                console.error("Error: Encryption key could not be retrieved.");
            }
        } else {
            console.error("Error: Session ID could not be retrieved.");
        }
    } catch (error) {
        console.error("Error initializing WebSocket:", error);
    }
}

initializeWebSocket();

async function decryptData(encryptedData, key) {
    const decodedData = window.atob(encryptedData);
    const bytes = new Uint8Array(decodedData.length);
    for (let i = 0; i < decodedData.length; i++) {
        bytes[i] = decodedData.charCodeAt(i);
    }

    const iv = bytes.slice(0, 12);
    const tag = bytes.slice(12, 28);
    const ciphertext = bytes.slice(28, bytes.length - 16);
    const salt = bytes.slice(bytes.length - 16, bytes.length);

    const cryptoKey = await deriveKey(key, salt);

    try {
        const decryptedContent = await window.crypto.subtle.decrypt(
            { name: "AES-GCM", iv: iv, additionalData: new Uint8Array(), tagLength: 128 },
            cryptoKey,
            ciphertext
        );
        const dec = new TextDecoder();
        return dec.decode(decryptedContent);
    } catch (err) {
        console.error("Decryption failed:", err);
        throw err;
    }
}

async function deriveKey(password, salt) {
    const encoder = new TextEncoder();
    const keyMaterial = await window.crypto.subtle.importKey(
        "raw",
        encoder.encode(password),
        { name: "PBKDF2" },
        false,
        ["deriveBits", "deriveKey"]
    );
    return window.crypto.subtle.deriveKey(
        {
            name: "PBKDF2",
            salt: salt,
            iterations: 16384,
            hash: 'SHA-256'
        },
        keyMaterial,
        { name: "AES-GCM", length: 256 },
        true,
        ["encrypt", "decrypt"]
    );
}

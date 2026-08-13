using System;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace FoundationPoseStreaming
{
    public enum FPSenderState
    {
        Stopped,
        Connecting,
        WaitingForRegistrationFrame,
        SendingRegistrationFrame,
        TrackingStream,
        Error
    }

    public sealed class FoundationPoseTcpSender : MonoBehaviour
    {
        [Header("PC Server")]
        public string host = "127.0.0.1";
        public int port = 5000;
        public bool connectOnStart = true;
        public int connectTimeoutMs = 3000;
        public int sendTimeoutMs = 3000;

        [Header("Logging")]
        public bool verboseLogging = true;

        readonly object gate = new object();
        readonly AutoResetEvent frameAvailable = new AutoResetEvent(false);

        Thread workerThread;
        TcpClient client;
        NetworkStream stream;
        volatile bool stopRequested;

        byte[] pendingRegistrationMessage;
        string pendingRegistrationFrameId;
        byte[] latestTrackingMessage;
        string latestTrackingFrameId;

        long sentFrameCount;
        long droppedTrackingFrameCount;
        FPSenderState state = FPSenderState.Stopped;
        string lastError;

        public FPSenderState State
        {
            get
            {
                lock (gate)
                {
                    return state;
                }
            }
        }

        public bool NeedsRegistrationFrame => State == FPSenderState.WaitingForRegistrationFrame;
        public bool IsTrackingStream => State == FPSenderState.TrackingStream;
        public long SentFrameCount => Interlocked.Read(ref sentFrameCount);
        public long DroppedTrackingFrameCount => Interlocked.Read(ref droppedTrackingFrameCount);
        public string LastError => lastError;

        void Start()
        {
            if (connectOnStart)
            {
                StartSender();
            }
        }

        void OnDestroy()
        {
            StopSender();
        }

        public void StartSender()
        {
            lock (gate)
            {
                if (workerThread != null && workerThread.IsAlive)
                {
                    return;
                }

                stopRequested = false;
                state = FPSenderState.Connecting;
                workerThread = new Thread(SendLoop)
                {
                    IsBackground = true,
                    Name = "FoundationPose TCP Sender"
                };
                workerThread.Start();
            }
        }

        public void StopSender()
        {
            stopRequested = true;
            CloseSocket();
            frameAvailable.Set();

            Thread threadToJoin;
            lock (gate)
            {
                threadToJoin = workerThread;
            }

            if (threadToJoin != null && threadToJoin.IsAlive)
            {
                threadToJoin.Join(1000);
            }

            lock (gate)
            {
                if (threadToJoin == null || !threadToJoin.IsAlive)
                {
                    workerThread = null;
                }
                else
                {
                    Debug.LogWarning("[FoundationPoseTcpSender] Sender thread did not stop within 1000 ms; keeping thread reference to prevent duplicate sender threads.");
                }

                pendingRegistrationMessage = null;
                pendingRegistrationFrameId = null;
                latestTrackingMessage = null;
                latestTrackingFrameId = null;
                if (state != FPSenderState.Error)
                {
                    state = FPSenderState.Stopped;
                }
            }
        }

        public bool EnqueueRegistrationFrame(FPEncodedFrame frame)
        {
            if (frame == null)
            {
                throw new ArgumentNullException(nameof(frame));
            }

            lock (gate)
            {
                if (state != FPSenderState.WaitingForRegistrationFrame)
                {
                    return false;
                }

                if (pendingRegistrationMessage != null)
                {
                    return false;
                }

                pendingRegistrationMessage = frame.message;
                pendingRegistrationFrameId = frame.header.frame_id;
                frameAvailable.Set();
                return true;
            }
        }

        public bool EnqueueTrackingFrame(FPEncodedFrame frame)
        {
            if (frame == null)
            {
                throw new ArgumentNullException(nameof(frame));
            }

            lock (gate)
            {
                if (state != FPSenderState.TrackingStream)
                {
                    return false;
                }

                if (latestTrackingMessage != null)
                {
                    Interlocked.Increment(ref droppedTrackingFrameCount);
                }

                latestTrackingMessage = frame.message;
                latestTrackingFrameId = frame.header.frame_id;
                frameAvailable.Set();
                return true;
            }
        }

        void SendLoop()
        {
            try
            {
                client = new TcpClient();
                client.NoDelay = true;
                client.SendTimeout = sendTimeoutMs;
                ConnectWithTimeout(client, host, port, connectTimeoutMs);
                stream = client.GetStream();

                lock (gate)
                {
                    state = FPSenderState.WaitingForRegistrationFrame;
                }

                Log($"Connected to FoundationPose server {host}:{port}");

                while (!stopRequested)
                {
                    byte[] message = null;
                    string frameId = null;
                    bool registrationFrame = false;

                    lock (gate)
                    {
                        if (pendingRegistrationMessage != null)
                        {
                            state = FPSenderState.SendingRegistrationFrame;
                            message = pendingRegistrationMessage;
                            frameId = pendingRegistrationFrameId;
                            pendingRegistrationMessage = null;
                            pendingRegistrationFrameId = null;
                            registrationFrame = true;
                        }
                        else if (state == FPSenderState.TrackingStream && latestTrackingMessage != null)
                        {
                            message = latestTrackingMessage;
                            frameId = latestTrackingFrameId;
                            latestTrackingMessage = null;
                            latestTrackingFrameId = null;
                        }
                    }

                    if (message == null)
                    {
                        frameAvailable.WaitOne(20);
                        continue;
                    }

                    DateTime start = DateTime.UtcNow;
                    stream.Write(message, 0, message.Length);
                    stream.Flush();
                    Interlocked.Increment(ref sentFrameCount);
                    double sendMs = (DateTime.UtcNow - start).TotalMilliseconds;

                    if (registrationFrame)
                    {
                        lock (gate)
                        {
                            if (!stopRequested)
                            {
                                state = FPSenderState.TrackingStream;
                            }
                        }
                        Log($"Sent REGISTER frame {frameId}, bytes={message.Length}, send_ms={sendMs:F2}");
                    }
                    else if (verboseLogging)
                    {
                        Log($"Sent TRACK frame {frameId}, bytes={message.Length}, send_ms={sendMs:F2}, dropped_tracking={DroppedTrackingFrameCount}");
                    }
                }
            }
            catch (Exception ex)
            {
                lastError = ex.ToString();
                lock (gate)
                {
                    state = FPSenderState.Error;
                }
                Debug.LogError($"FoundationPose sender error: {ex}");
            }
            finally
            {
                CloseSocket();
            }
        }

        void CloseSocket()
        {
            try
            {
                stream?.Close();
            }
            catch
            {
                // Ignore close failures during teardown.
            }

            try
            {
                client?.Close();
            }
            catch
            {
                // Ignore close failures during teardown.
            }

            stream = null;
            client = null;
        }

        static void ConnectWithTimeout(TcpClient tcpClient, string targetHost, int targetPort, int timeoutMs)
        {
            IAsyncResult result = tcpClient.BeginConnect(targetHost, targetPort, null, null);
            bool connected = result.AsyncWaitHandle.WaitOne(Math.Max(1, timeoutMs));
            if (!connected)
            {
                try
                {
                    tcpClient.Close();
                }
                catch
                {
                    // Ignore close failures on timeout.
                }

                throw new TimeoutException($"Timed out connecting to {targetHost}:{targetPort} after {timeoutMs} ms");
            }

            tcpClient.EndConnect(result);
        }

        void Log(string message)
        {
            if (verboseLogging)
            {
                Debug.Log($"[FoundationPoseTcpSender] {message}");
            }
        }
    }
}

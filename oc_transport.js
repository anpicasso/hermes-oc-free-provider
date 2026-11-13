// Bun transport for oc-free-provider.
//
// OpenCode's free tier only answers requests that leave from Bun's TLS stack,
// so Hermes hands the request to this process instead of urllib.
// stdin: {"url", "headers", "body"}   stdout: {"status":N}\n + raw response bytes
const input = JSON.parse(await Bun.stdin.text());
const res = await fetch(input.url, {
  method: "POST",
  headers: input.headers,
  body: input.body,
});
process.stdout.write(JSON.stringify({ status: res.status }) + "\n");
if (res.body) {
  for await (const chunk of res.body) process.stdout.write(chunk);
}
process.exit(0);

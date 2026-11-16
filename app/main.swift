import Cocoa
import WebKit
let home = FileManager.default.homeDirectoryForCurrentUser.path
class D: NSObject, NSApplicationDelegate, WKUIDelegate, NSWindowDelegate {
  var win: NSWindow!; var web: WKWebView!; var server: Process?
  func applicationDidFinishLaunching(_ n: Notification) {
    startServer()
    // keep the server alive: if it stops answering twice in a row, start a fresh one
    Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in self.healthCheck() }
    win = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 820),
                   styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
    win.titlebarAppearsTransparent = true; win.titleVisibility = .hidden; win.title = "Rivet"; win.minSize = NSSize(width: 520, height: 420)
    win.center(); win.setFrameAutosaveName("RivetMain"); win.delegate = self
    let cfg = WKWebViewConfiguration()
    web = WKWebView(frame: win.contentView!.bounds, configuration: cfg); web.autoresizingMask = [.width, .height]
    web.uiDelegate = self; web.setValue(false, forKey: "drawsBackground")
    win.contentView!.addSubview(web); win.makeKeyAndOrderFront(nil)
    tryLoad()
    let m = NSMenu()
    let ai = NSMenuItem(); m.addItem(ai); let am = NSMenu()
    am.addItem(withTitle: "About Rivet", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
    am.addItem(.separator()); am.addItem(withTitle: "Hide Rivet", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
    am.addItem(withTitle: "Quit Rivet", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"); ai.submenu = am
    let ei = NSMenuItem(); m.addItem(ei); let em = NSMenu(title: "Edit")
    for (t, s, k) in [("Undo","undo:","z"),("Redo","redo:","Z"),("Cut","cut:","x"),("Copy","copy:","c"),("Paste","paste:","v"),("Select All","selectAll:","a")] {
      em.addItem(withTitle: t, action: Selector(s), keyEquivalent: k) }
    ei.submenu = em
    let wi = NSMenuItem(); m.addItem(wi); let wm = NSMenu(title: "Window")
    wm.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
    wm.addItem(withTitle: "Reload", action: #selector(reload), keyEquivalent: "r"); wi.submenu = wm
    NSApp.mainMenu = m; NSApp.activate(ignoringOtherApps: true)
  }
  @objc func reload() { web.reload() }
  var misses = 0
  func startServer() {
    if let old = server, old.isRunning { old.terminate() }
    let p = Process()
    p.executableURL = URL(fileURLWithPath: home + "/LocalVoice/.venv/bin/python")
    p.arguments = [home + "/Rivet/server.py"]
    let log = home + "/Rivet/server.log"
    FileManager.default.createFile(atPath: log, contents: nil)
    if let h = FileHandle(forWritingAtPath: log) { p.standardOutput = h; p.standardError = h }
    try? p.run(); server = p
  }
  func healthCheck() {
    var r = URLRequest(url: URL(string: "http://127.0.0.1:4480/state")!); r.timeoutInterval = 12
    r.cachePolicy = .reloadIgnoringLocalCacheData
    URLSession.shared.dataTask(with: r) { _, resp, err in DispatchQueue.main.async {
      if (resp as? HTTPURLResponse)?.statusCode == 200 { self.misses = 0; return }
      // Only a DEAD server (connection refused / can't connect) counts as a miss.
      // A timeout means the server is alive but busy streaming a heavy reply — never restart for that.
      let code = (err as NSError?)?.code
      let dead = code == NSURLErrorCannotConnectToHost || code == NSURLErrorCannotFindHost
              || code == NSURLErrorNetworkConnectionLost || code == NSURLErrorNotConnectedToInternet
      if !dead { self.misses = 0; return }                 // busy/slow, or any non-fatal error: leave it alone
      self.misses += 1
      if self.misses >= 3 { self.misses = 0; self.startServer()
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { self.web.reload() } }
    } }.resume()
  }
  func tryLoad() {
    var r = URLRequest(url: URL(string: "http://127.0.0.1:4480/state")!); r.timeoutInterval = 1
    URLSession.shared.dataTask(with: r) { _, resp, _ in DispatchQueue.main.async {
      if resp != nil { self.web.load(URLRequest(url: URL(string: "http://127.0.0.1:4480/")!)) }
      else { DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { self.tryLoad() } } } }.resume()
  }
  // <input type="file"> (attach images)
  func webView(_ w: WKWebView, runOpenPanelWith p: WKOpenPanelParameters, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
    let o = NSOpenPanel(); o.allowsMultipleSelection = p.allowsMultipleSelection; o.canChooseDirectories = false
    o.allowedContentTypes = [.image]; o.beginSheetModal(for: win) { completionHandler($0 == .OK ? o.urls : nil) }
  }
  // confirm() and prompt() from the page (delete / rename chats)
  func webView(_ w: WKWebView, runJavaScriptConfirmPanelWithMessage msg: String, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
    let a = NSAlert(); a.messageText = msg; a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
    a.beginSheetModal(for: win) { completionHandler($0 == .alertFirstButtonReturn) }
  }
  func webView(_ w: WKWebView, runJavaScriptTextInputPanelWithPrompt msg: String, defaultText: String?, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping (String?) -> Void) {
    let a = NSAlert(); a.messageText = msg; a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
    let t = NSTextField(frame: NSRect(x: 0, y: 0, width: 280, height: 24)); t.stringValue = defaultText ?? ""; a.accessoryView = t
    a.window.initialFirstResponder = t
    a.beginSheetModal(for: win) { completionHandler($0 == .alertFirstButtonReturn ? t.stringValue : nil) }
  }
  // links open in the real browser
  func webView(_ w: WKWebView, createWebViewWith c: WKWebViewConfiguration, for a: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
    if let u = a.request.url { NSWorkspace.shared.open(u) }; return nil
  }
  func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
  func applicationWillTerminate(_ n: Notification) { server?.terminate() }
}
let app = NSApplication.shared; let d = D(); app.delegate = d; app.setActivationPolicy(.regular); app.run()

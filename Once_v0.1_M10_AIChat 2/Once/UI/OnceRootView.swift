import SwiftUI
import UIKit

struct OnceRootView: View {
    private let messages = [
        "你好，我是Once。我负责陪你把你脑子里的画面，一点点真的变成属于你的动漫世界。",
        "你不用会画，也不用一开始就知道该怎么表达。你可以说、可以画、可以乱来。我们一起试，直到那个画面越来越像你想的那一个。",
        "我不会替你创作，也不会假装什么都懂。这个世界还是你的。我只负责陪你，把它做出来。"
    ]

    @State private var orbVisible = false
    @State private var hasStarted = false
    @State private var currentMessageIndex = 0
    @State private var visibleCharacterIndexes: Set<Int> = []
    @State private var isAnimatingText = false

    @State private var showNamePrompt = false
    @State private var namePromptVisible = false
    @State private var nickname = ""

    @State private var isEnteringChat = false
    @State private var chatVisible = false
    @State private var introOpacity: Double = 1
    @State private var introBlur: CGFloat = 0

    @State private var chatText = ""
    @State private var chatMessages: [OnceLocalChatMessage] = []
    @StateObject private var chatService = OnceChatService()

    @FocusState private var isNameFocused: Bool
    @FocusState private var isChatFocused: Bool
    @AppStorage("oncePreferredName") private var preferredName = ""

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.white.ignoresSafeArea()

                if !chatVisible {
                    introView(in: proxy)
                        .opacity(introOpacity)
                        .blur(radius: introBlur)
                }

                if chatVisible {
                    OnceConversationView(
                        text: $chatText,
                        messages: $chatMessages,
                        isFocused: $isChatFocused,
                        aiService: chatService,
                        preferredName: preferredName
                    )
                    .transition(.opacity)
                }
            }
            .onAppear {
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(250))
                    withAnimation(.easeOut(duration: 0.95)) {
                        orbVisible = true
                    }
                }
            }
        }
        .preferredColorScheme(.light)
    }

    @ViewBuilder
    private func introView(in proxy: GeometryProxy) -> some View {
        let stableHeight = max(proxy.size.height, UIScreen.main.bounds.height)
        let stableWidth = max(proxy.size.width, UIScreen.main.bounds.width)
        let initialOrbY = stableHeight * 0.50
        let activeOrbY = stableHeight * 0.27
        let activeOrbSize: CGFloat = 68
        let textTop = activeOrbY + (activeOrbSize / 2) + 38
        let namePromptY = isNameFocused ? stableHeight * 0.47 : stableHeight * 0.67

        ZStack {
            Color.white.ignoresSafeArea()

            if !isEnteringChat {
                VStack {
                    HStack {
                        Button(action: skipToChat) {
                            Text("跳过")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(Color.black.opacity(0.50))
                                .padding(.horizontal, 14)
                                .padding(.vertical, 8)
                                .background(
                                    Capsule()
                                        .fill(Color.black.opacity(0.055))
                                )
                                .overlay {
                                    Capsule()
                                        .stroke(Color.black.opacity(0.07), lineWidth: 0.7)
                                }
                        }
                        .buttonStyle(.plain)

                        Spacer()
                    }
                    Spacer()
                }
                .padding(.leading, max(proxy.safeAreaInsets.leading, 16) + 2)
                .padding(.top, max(proxy.safeAreaInsets.top, 10) + 2)
                .zIndex(30)
            }

            Circle()
                .fill(.black)
                .frame(
                    width: hasStarted ? activeOrbSize : 92,
                    height: hasStarted ? activeOrbSize : 92
                )
                .opacity(orbVisible ? 1 : 0)
                .blur(radius: orbVisible ? 0 : 10)
                .scaleEffect(orbVisible ? 1 : 0.82)
                .position(
                    x: proxy.size.width / 2,
                    y: hasStarted ? activeOrbY : initialOrbY
                )
                .animation(.spring(response: 0.58, dampingFraction: 0.84), value: hasStarted)

            if hasStarted {
                GlyphRevealText(
                    text: messages[currentMessageIndex],
                    visibleIndexes: visibleCharacterIndexes
                )
                .frame(maxWidth: min(stableWidth - 54, 430))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .padding(.top, textTop)
            }

            if !showNamePrompt && !isEnteringChat {
                Color.clear
                    .contentShape(Rectangle())
                    .ignoresSafeArea()
                    .onTapGesture {
                        handleIntroTap()
                    }
            }

            if showNamePrompt && !isEnteringChat {
                Color.clear
                    .contentShape(Rectangle())
                    .ignoresSafeArea()
                    .onTapGesture {
                        finishNameAndEnterChatIfPossible()
                    }

                NamePromptCard(
                    nickname: $nickname,
                    isFocused: $isNameFocused,
                    onSubmit: finishNameAndEnterChatIfPossible
                )
                .frame(maxWidth: min(stableWidth - 44, 360))
                .position(x: stableWidth / 2, y: namePromptY)
                .opacity(namePromptVisible ? 1 : 0)
                .blur(radius: namePromptVisible ? 0 : 12)
                .scaleEffect(namePromptVisible ? 1 : 0.985)
                .animation(.easeInOut(duration: 0.28), value: isNameFocused)
            }
        }
    }

    private func handleIntroTap() {
        guard orbVisible, !isAnimatingText, !showNamePrompt, !isEnteringChat else { return }

        if !hasStarted {
            let generator = UIImpactFeedbackGenerator(style: .light)
            generator.prepare()
            generator.impactOccurred(intensity: 0.55)

            withAnimation(.spring(response: 0.58, dampingFraction: 0.84)) {
                hasStarted = true
            }

            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(500))
                await revealCurrentMessage()
            }
            return
        }

        guard currentMessageIndex < messages.count - 1 else {
            Task { @MainActor in
                await presentNamePrompt()
            }
            return
        }

        Task { @MainActor in
            await hideCurrentMessage()
            currentMessageIndex += 1
            try? await Task.sleep(for: .milliseconds(220))
            await revealCurrentMessage()
        }
    }

    @MainActor
    private func revealCurrentMessage() async {
        isAnimatingText = true
        visibleCharacterIndexes.removeAll()

        let count = Array(messages[currentMessageIndex]).count
        guard count > 0 else {
            isAnimatingText = false
            return
        }

        for index in 0..<count {
            withAnimation(.easeOut(duration: 0.32)) {
                _ = visibleCharacterIndexes.insert(index)
            }
            try? await Task.sleep(for: .milliseconds(36))
        }

        try? await Task.sleep(for: .milliseconds(180))
        isAnimatingText = false

        if currentMessageIndex == messages.count - 1 {
            try? await Task.sleep(for: .milliseconds(420))
            await presentNamePrompt()
        }
    }

    @MainActor
    private func hideCurrentMessage() async {
        isAnimatingText = true
        let count = Array(messages[currentMessageIndex]).count

        guard count > 0 else {
            isAnimatingText = false
            return
        }

        for index in stride(from: count - 1, through: 0, by: -1) {
            withAnimation(.easeIn(duration: 0.25)) {
                visibleCharacterIndexes.remove(index)
            }
            try? await Task.sleep(for: .milliseconds(26))
        }

        visibleCharacterIndexes.removeAll()
        isAnimatingText = false
    }

    @MainActor
    private func presentNamePrompt() async {
        guard !showNamePrompt, !isEnteringChat else { return }
        showNamePrompt = true
        namePromptVisible = false

        try? await Task.sleep(for: .milliseconds(40))
        withAnimation(.easeOut(duration: 0.62)) {
            namePromptVisible = true
        }
    }

    private func finishNameAndEnterChatIfPossible() {
        let cleaned = nickname.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else {
            isNameFocused = false
            return
        }

        preferredName = cleaned
        isNameFocused = false
        enterChat()
    }

    private func skipToChat() {
        guard !isEnteringChat else { return }
        isNameFocused = false
        enterChat()
    }

    private func enterChat() {
        guard !isEnteringChat else { return }
        isEnteringChat = true

        Task { @MainActor in
            withAnimation(.easeInOut(duration: 0.72)) {
                namePromptVisible = false
                introOpacity = 0
                introBlur = 14
            }

            try? await Task.sleep(for: .milliseconds(760))
            chatVisible = true

            withAnimation(.easeOut(duration: 0.50)) {
                introOpacity = 0
                introBlur = 0
            }

            try? await Task.sleep(for: .milliseconds(180))
            isChatFocused = true
        }
    }
}

private enum OnceChatRole: String, Codable {
    case user
    case assistant
}

private struct OnceLocalChatMessage: Identifiable, Codable, Equatable {
    let id: UUID
    let role: OnceChatRole
    let text: String

    init(id: UUID = UUID(), role: OnceChatRole, text: String) {
        self.id = id
        self.role = role
        self.text = text
    }
}

private enum OnceChatServiceError: LocalizedError {
    case missingBaseURL
    case invalidBaseURL
    case invalidResponse
    case backend(String)

    var errorDescription: String? {
        switch self {
        case .missingBaseURL:
            return "还没连接 Render。点右上角设置，填入 Once 后端地址。"
        case .invalidBaseURL:
            return "Render 地址看起来不对。"
        case .invalidResponse:
            return "后端有响应，但 Once 没读懂返回内容。"
        case .backend(let message):
            return message
        }
    }
}

@MainActor
private final class OnceChatService: ObservableObject {
    @Published private(set) var isSending = false
    @Published private(set) var statusText = ""

    private let baseURLKey = "once.backend.baseURL"

    var baseURL: String {
        UserDefaults.standard.string(forKey: baseURLKey) ?? ""
    }

    func updateBaseURL(_ value: String) {
        let cleaned = value.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        UserDefaults.standard.set(cleaned, forKey: baseURLKey)
    }

    func reply(to messages: [OnceLocalChatMessage], preferredName: String) async throws -> String {
        let base = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !base.isEmpty else { throw OnceChatServiceError.missingBaseURL }
        guard let url = URL(string: base + "/once/chat") else { throw OnceChatServiceError.invalidBaseURL }

        isSending = true
        statusText = "Once 正在想…"
        defer {
            isSending = false
            statusText = ""
        }

        let recent = messages.suffix(40).map { message in
            [
                "role": message.role.rawValue,
                "content": message.text
            ]
        }

        let payload: [String: Any] = [
            "nickname": preferredName,
            "messages": recent
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 90
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw OnceChatServiceError.invalidResponse
        }

        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]

        guard (200...299).contains(http.statusCode) else {
            let message = (json?["detail"] as? String)
                ?? (json?["message"] as? String)
                ?? (json?["error"] as? String)
                ?? "Render 请求失败（\(http.statusCode)）"
            throw OnceChatServiceError.backend(message)
        }

        guard let reply = json?["reply"] as? String, !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OnceChatServiceError.invalidResponse
        }

        return reply
    }
}

private struct OnceConversationView: View {
    @Binding var text: String
    @Binding var messages: [OnceLocalChatMessage]
    var isFocused: FocusState<Bool>.Binding
    @ObservedObject var aiService: OnceChatService
    let preferredName: String

    @State private var showSettings = false
    @State private var settingsURL = ""

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.white.ignoresSafeArea()

                VStack(spacing: 0) {
                    HStack(spacing: 9) {
                        Circle()
                            .fill(Color.black)
                            .frame(width: 34, height: 34)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Once")
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(Color.black.opacity(0.86))

                            if aiService.isSending {
                                Text("正在想")
                                    .font(.system(size: 10.5, weight: .medium))
                                    .foregroundStyle(Color.black.opacity(0.34))
                            }
                        }

                        Spacer()

                        Button {
                            settingsURL = aiService.baseURL
                            showSettings = true
                        } label: {
                            Image(systemName: "gearshape")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(Color.black.opacity(0.48))
                                .frame(width: 34, height: 34)
                                .background(
                                    Circle()
                                        .fill(Color.black.opacity(0.045))
                                )
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("AI 连接设置")
                    }
                    .padding(.horizontal, 18)
                    .padding(.top, max(proxy.safeAreaInsets.top, 10) + 4)
                    .padding(.bottom, 10)

                    ScrollViewReader { scrollProxy in
                        ScrollView(.vertical, showsIndicators: false) {
                            LazyVStack(spacing: 10) {
                                if messages.isEmpty {
                                    VStack(spacing: 7) {
                                        Text("跟我聊聊你现在脑子里的东西。")
                                            .font(.system(size: 15, weight: .semibold))
                                            .foregroundStyle(Color.black.opacity(0.56))
                                        Text("可以只是一个画面，也可以是一整个故事。")
                                            .font(.system(size: 12.5, weight: .regular))
                                            .foregroundStyle(Color.black.opacity(0.28))
                                    }
                                    .padding(.top, 42)
                                }

                                ForEach(messages) { message in
                                    OnceChatBubble(message: message)
                                        .id(message.id)
                                }

                                if aiService.isSending {
                                    HStack {
                                        HStack(spacing: 5) {
                                            Circle().frame(width: 5, height: 5)
                                            Circle().frame(width: 5, height: 5)
                                            Circle().frame(width: 5, height: 5)
                                        }
                                        .foregroundStyle(Color.black.opacity(0.36))
                                        .padding(.horizontal, 13)
                                        .padding(.vertical, 11)
                                        .background(
                                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                                .fill(Color.black.opacity(0.055))
                                        )
                                        Spacer(minLength: proxy.size.width * 0.22)
                                    }
                                    .id("once-thinking")
                                }
                            }
                            .padding(.horizontal, 18)
                            .padding(.vertical, 8)
                        }
                        .onChange(of: messages.count) { _ in
                            scrollToBottom(using: scrollProxy)
                        }
                        .onChange(of: aiService.isSending) { _ in
                            scrollToBottom(using: scrollProxy)
                        }
                    }
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                OnceChatComposer(
                    text: $text,
                    isFocused: isFocused,
                    isSending: aiService.isSending,
                    onSend: sendMessage
                )
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, isFocused.wrappedValue ? 0 : max(proxy.safeAreaInsets.bottom, 8))
                .background(Color.white)
            }
        }
        .sheet(isPresented: $showSettings) {
            OnceBackendSettingsSheet(
                baseURL: $settingsURL,
                onSave: {
                    aiService.updateBaseURL(settingsURL)
                    showSettings = false
                }
            )
            .presentationDetents([.height(250)])
        }
    }

    private func scrollToBottom(using proxy: ScrollViewProxy) {
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(80))
            if aiService.isSending {
                withAnimation(.easeOut(duration: 0.22)) {
                    proxy.scrollTo("once-thinking", anchor: .bottom)
                }
            } else if let last = messages.last {
                withAnimation(.easeOut(duration: 0.22)) {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private func sendMessage() {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty, !aiService.isSending else { return }

        let userMessage = OnceLocalChatMessage(role: .user, text: cleaned)
        messages.append(userMessage)
        text = ""

        let snapshot = messages
        Task { @MainActor in
            do {
                let reply = try await aiService.reply(to: snapshot, preferredName: preferredName)
                messages.append(OnceLocalChatMessage(role: .assistant, text: reply))
            } catch {
                messages.append(
                    OnceLocalChatMessage(
                        role: .assistant,
                        text: "我这边还没连上。\(error.localizedDescription)"
                    )
                )
            }
        }
    }
}

private struct OnceChatBubble: View {
    let message: OnceLocalChatMessage

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 58)
            }

            Text(message.text)
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(message.role == .user ? Color.white : Color.black.opacity(0.84))
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(message.role == .user ? Color.black : Color.black.opacity(0.055))
                )

            if message.role == .assistant {
                Spacer(minLength: 58)
            }
        }
    }
}

private struct OnceBackendSettingsSheet: View {
    @Binding var baseURL: String
    let onSave: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("连接 Once AI")
                        .font(.system(size: 18, weight: .semibold))
                    Text("填你部署到 Render 的后端地址")
                        .font(.system(size: 12, weight: .regular))
                        .foregroundStyle(Color.black.opacity(0.40))
                }
                Spacer()
                Button("关闭") { dismiss() }
                    .buttonStyle(.plain)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Color.black.opacity(0.48))
            }

            TextField("https://your-once.onrender.com", text: $baseURL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled(true)
                .font(.system(size: 14, weight: .regular))
                .padding(.horizontal, 13)
                .frame(height: 44)
                .background(
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(Color.black.opacity(0.05))
                )

            Button(action: onSave) {
                Text("保存")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Color.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 42)
                    .background(
                        RoundedRectangle(cornerRadius: 13, style: .continuous)
                            .fill(Color.black)
                    )
            }
            .buttonStyle(.plain)
        }
        .padding(20)
    }
}

private struct OnceChatComposer: View {
    @Binding var text: String
    var isFocused: FocusState<Bool>.Binding
    let isSending: Bool
    let onSend: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("跟 Once 说点什么…", text: $text, axis: .vertical)
                .focused(isFocused)
                .lineLimit(1...5)
                .font(.system(size: 14.5, weight: .regular))
                .foregroundStyle(Color.black.opacity(0.88))
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .fill(Color.black.opacity(0.045))
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Color.black.opacity(0.07), lineWidth: 0.8)
                }
                .submitLabel(.send)
                .onSubmit {
                    onSend()
                }

            Button(action: onSend) {
                Image(systemName: isSending ? "ellipsis" : "arrow.up")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(Color.white)
                    .frame(width: 42, height: 42)
                    .background(
                        Circle()
                            .fill(canSend ? Color.black : Color.black.opacity(0.22))
                    )
            }
            .buttonStyle(.plain)
            .disabled(!canSend)
        }
    }

    private var canSend: Bool {
        !isSending && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

private struct GlyphRevealText: View {
    let text: String
    let visibleIndexes: Set<Int>

    var body: some View {
        let characters = Array(text)

        GlyphFlowLayout(horizontalSpacing: 0, verticalSpacing: 7) {
            ForEach(Array(characters.enumerated()), id: \.offset) { index, character in
                Text(String(character))
                    .font(.system(size: 19, weight: .semibold))
                    .foregroundStyle(Color.black.opacity(visibleIndexes.contains(index) ? 0.90 : 0.001))
                    .blur(radius: visibleIndexes.contains(index) ? 0 : 8)
                    .fixedSize()
            }
        }
        .frame(maxWidth: .infinity)
    }
}

private struct GlyphFlowLayout: Layout {
    let horizontalSpacing: CGFloat
    let verticalSpacing: CGFloat

    struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        let maxWidth = proposal.width ?? UIScreen.main.bounds.width
        let rows = makeRows(maxWidth: maxWidth, subviews: subviews)
        let height = rows.enumerated().reduce(CGFloat.zero) { partial, item in
            partial + item.element.height + (item.offset == rows.count - 1 ? 0 : verticalSpacing)
        }
        return CGSize(width: maxWidth, height: height)
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let rows = makeRows(maxWidth: bounds.width, subviews: subviews)
        var y = bounds.minY

        for row in rows {
            var x = bounds.midX - row.width / 2
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(
                    at: CGPoint(x: x, y: y + (row.height - size.height) / 2),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(size)
                )
                x += size.width + horizontalSpacing
            }
            y += row.height + verticalSpacing
        }
    }

    private func makeRows(maxWidth: CGFloat, subviews: Subviews) -> [Row] {
        var rows: [Row] = []
        var current = Row()

        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let proposedWidth = current.indices.isEmpty
                ? size.width
                : current.width + horizontalSpacing + size.width

            if proposedWidth > maxWidth, !current.indices.isEmpty {
                rows.append(current)
                current = Row(indices: [index], width: size.width, height: size.height)
            } else {
                current.indices.append(index)
                current.width = proposedWidth
                current.height = max(current.height, size.height)
            }
        }

        if !current.indices.isEmpty {
            rows.append(current)
        }
        return rows
    }
}

private struct NamePromptCard: View {
    @Binding var nickname: String
    var isFocused: FocusState<Bool>.Binding
    let onSubmit: () -> Void

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) {
                nameLabel
                nameField
                    .frame(width: 128)
            }

            VStack(spacing: 10) {
                nameLabel
                nameField
                    .frame(maxWidth: 220)
            }
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 15)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Color.white)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color.black.opacity(0.07), lineWidth: 0.8)
        }
        .shadow(color: Color.black.opacity(0.07), radius: 22, x: 0, y: 8)
    }

    private var nameLabel: some View {
        Text("你好，以后我就叫你")
            .font(.system(size: 16, weight: .semibold))
            .foregroundStyle(Color.black.opacity(0.88))
    }

    private var nameField: some View {
        TextField("名字", text: $nickname)
            .focused(isFocused)
            .submitLabel(.done)
            .onSubmit(onSubmit)
            .font(.system(size: 16, weight: .semibold))
            .foregroundStyle(Color.black.opacity(0.88))
            .multilineTextAlignment(.center)
            .padding(.horizontal, 12)
            .frame(height: 42)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.black.opacity(0.055))
            )
            .overlay {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Color.black.opacity(0.08), lineWidth: 0.8)
            }
    }
}

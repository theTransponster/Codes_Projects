import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class PanelControles extends JPanel{

	private PanelHipoFlexion phf;
	private JButton btExpediente,
					btRegresar;
	private JSlider sMatizmin,
					sMatizmax,
					sSatmin,
					sSatmax,
					sValmin,
					sValmax;
	private JTextArea texto;
	private JLabel lMatmax,
				   lMatmin,
				   lSatmax,
				   lSatmin,
				   lValmax,
				   lValmin;
	private String movimiento;
	public boolean movs;
	public JFileChooser fc;
	public File f;
	private ArrayList<String> mov, 
							  movimientos; 
	
	
	               
	public PanelControles(PanelHipoFlexion phf){
		super();
		this.phf=phf;
		this.movs=true;
		this.mov=new ArrayList<String>();
		this.movimiento="Esperando movimiento...";
		this.fc=new JFileChooser();
		this.setPreferredSize(new Dimension(480,570));
		this.setLayout(null);
		this.texto=new JTextArea("INSTRUCCIONES"+"\n"+"1. Coloque el codo derecho a la altura del círculo azul"+
								"\n"+"2. Ajuste las barras de abajo hasta que se detecte su objeto"+
								"\n"+"3. Posicione su objeto en el círculo verde para iniciar"+"\n"+
								"4. Mueva el objeto hasta alcanzar los círculos rojos"+"\n"+"5. Realice las repeticiones necesarias"+"\n"+
								"6. Haga click en Generar Expediente para guardar su avance");
		this.texto.setFont(new Font("TimesRoman", Font.BOLD,13));
		this.texto.setForeground(Color.BLACK);
		this.add(this.texto);
		this.texto.setBounds(20, 15, 410, 150);
		
		this.btRegresar=new JButton("Regresar");
		this.add(this.btRegresar);
		this.btRegresar.setBounds(320,530,100,30);
		this.btRegresar.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				PanelControles.this.phf.vh.dispose();
				VentanaPrincipal v=new VentanaPrincipal();
			}
		});
		this.btExpediente=new JButton("Generar Expediente");
		this.add(this.btExpediente);
		this.btExpediente.setBounds(80, 530, 180, 30);
		this.btExpediente.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				movs=false;
				fc.showSaveDialog(null);
				f=fc.getSelectedFile();
				PrintWriter pw;
				String[] datos=phf.getDatos();
				ArrayList<String> resumen=PanelControles.this.lista();
				try {
					pw=new PrintWriter(new FileWriter(f+".txt"));
					pw.println("Expediente");
					pw.println(" ");
					pw.println("Nombre del paciente: "+datos[0]);
					pw.println("Edad: "+ datos[1]);
					pw.println("Terapia Número: "+datos[2]);
					pw.println("Fecha: "+datos[3]);
					pw.println(" ");
					pw.println("Resumen de Terapia y ángulos alcanzados: ");
					pw.println(resumen.get(0));
					for(int j=1;j<(resumen.size()-1);j++){
						if(resumen.get(j)!=resumen.get(j-1)){
							pw.println(resumen.get(j));
						}
					}
					pw.close();
				} catch (IOException e1) {
					e1.printStackTrace();
				}
			}
		});
		this.lMatmin=new JLabel("Matiz mínimo");
		this.lMatmin.setFont(new Font("TimesRoman", Font.BOLD,11));
		this.lMatmin.setForeground(Color.BLACK);
		this.add(this.lMatmin);
		this.lMatmin.setBounds(20, 200, 100, 20);
		this.sMatizmin=new JSlider(JSlider.HORIZONTAL,0,180,0);
		this.sMatizmin.setMajorTickSpacing(20);
		this.sMatizmin.setMinorTickSpacing(20);
		this.sMatizmin.setPaintLabels(true);
		this.sMatizmin.setPaintTicks(true);
		this.sMatizmin.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				phf.setMatMin(sMatizmin.getValue());	
			}
		});
		this.add(this.sMatizmin);
		this.sMatizmin.setBounds(20, 215, 220, 50);
		
		this.lMatmax=new JLabel("Matiz máximo");
		this.lMatmax.setFont(new Font("TimesRoman", Font.BOLD,11));
		this.lMatmax.setForeground(Color.BLACK);
		this.add(this.lMatmax);
		this.lMatmax.setBounds(260, 200, 100, 20);
		this.sMatizmax=new JSlider(JSlider.HORIZONTAL,0,180,0);
		this.sMatizmax.setMajorTickSpacing(20);
		this.sMatizmax.setMinorTickSpacing(20);
		this.sMatizmax.setPaintLabels(true);
		this.sMatizmax.setPaintTicks(true);
		this.sMatizmax.addChangeListener(new ChangeListener() {	
			public void stateChanged(ChangeEvent e) {
				phf.setMatMax(sMatizmax.getValue());
			}
		});
		this.add(this.sMatizmax);
		this.sMatizmax.setBounds(260, 215, 200, 50);
	
		this.lSatmin=new JLabel("Saturación mínima");
		this.lSatmin.setFont(new Font("TimesRoman", Font.BOLD,12));
		this.lSatmin.setForeground(Color.BLACK);
		this.add(this.lSatmin);
		this.lSatmin.setBounds(20, 280, 110, 20);
		this.sSatmin=new JSlider(JSlider.HORIZONTAL,0,255,48);
		this.sSatmin.setMajorTickSpacing(20);
		this.sSatmin.setMinorTickSpacing(20);
		this.sSatmin.setPaintLabels(true);
		this.sSatmin.setPaintTicks(true);
		this.sSatmin.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				phf.setSatMin(sSatmin.getValue());
			}
		});
		this.add(this.sSatmin);
		this.sSatmin.setBounds(20, 295, 220, 50);
	
		this.lSatmax=new JLabel("Saturación máxima");
		this.lSatmax.setFont(new Font("TimesRoman", Font.BOLD,12));
		this.lSatmax.setForeground(Color.BLACK);
		this.add(this.lSatmax);
		this.lSatmax.setBounds(260, 280, 110, 20);
		this.sSatmax=new JSlider(JSlider.HORIZONTAL,0,255,200);
		this.sSatmax.setMajorTickSpacing(20);
		this.sSatmax.setMinorTickSpacing(20);
		this.sSatmax.setPaintLabels(true);
		this.sSatmax.setPaintTicks(true);
		this.sSatmax.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				phf.setSatMax(sSatmax.getValue());
			}
		});
		this.add(this.sSatmax);
		this.sSatmax.setBounds(250, 295, 220, 50);
		
		this.lValmin=new JLabel("Valor mínimo");
		this.lValmin.setFont(new Font("TimesRoman", Font.BOLD,12));
		this.lValmin.setForeground(Color.BLACK);
		this.add(this.lValmin);
		this.lValmin.setBounds(20, 350, 110, 20);
		this.sValmin=new JSlider(JSlider.HORIZONTAL,0,255,80);
		this.sValmin.setMajorTickSpacing(20);
		this.sValmin.setMinorTickSpacing(20);
		this.sValmin.setPaintLabels(true);
		this.sValmin.setPaintTicks(true);
		this.sValmin.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				phf.setValmin(sValmin.getValue());	
			}
		});
		this.add(this.sValmin);
		this.sValmin.setBounds(20, 368, 220, 50);
		
		this.lValmax=new JLabel("Valor máximo");
		this.lValmax.setFont(new Font("TimesRoman", Font.BOLD,12));
		this.lValmax.setForeground(Color.BLACK);
		this.add(this.lValmax);
		this.lValmax.setBounds(260, 350, 110, 20);
		this.sValmax=new JSlider(JSlider.HORIZONTAL,0,255,255);
		this.sValmax.setMajorTickSpacing(20);
		this.sValmax.setMinorTickSpacing(20);
		this.sValmax.setPaintLabels(true);
		this.sValmax.setPaintTicks(true);
		this.sValmax.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				phf.setValMax(sValmax.getValue());
			}
		});
		this.add(this.sValmax);
		this.sValmax.setBounds(250, 368, 220, 50);	
	}
	public void setMov(String mov){
		this.movimiento=mov;
		repaint();
	}

	public ArrayList<String> lista(){
		return this.mov;
	}
	public ArrayList<String> listaMovs(String mov){
		this.movimiento=mov;
		ArrayList<String> movimientos=new ArrayList<String>();
			this.mov.add(this.movimiento);
			System.out.println(this.movimiento);
		lista();
		return this.mov;
	}
	
	public void paintComponent(Graphics g){
		super.paintComponent(g);
		g.setColor(Color.blue);
		g.setFont(new Font("TimesRoman", Font.BOLD,15));
		g.drawString(this.movimiento, 180, 475);
	}
	
	
}
